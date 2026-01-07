import copy
import logging
from typing import List, Optional

import torch

from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.layers.utils.logprob import get_token_ids_logprobs, get_top_logprobs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

if is_cuda() or is_hip():
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob


logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


def _top_k_renorm_prob_fallback(
    probs: torch.Tensor, top_ks: torch.Tensor
) -> torch.Tensor:
    # Fallback for non-CUDA environments; keep it simple.
    max_k = int(top_ks.max().item())
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    mask = torch.arange(max_k, device=probs.device).view(1, -1)
    keep_mask = mask < top_ks.view(-1, 1)
    sorted_probs = sorted_probs[:, :max_k]
    sorted_idx = sorted_idx[:, :max_k]
    sorted_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
    renorm = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sorted_probs = sorted_probs / renorm
    out = torch.zeros_like(probs)
    out.scatter_(1, sorted_idx, sorted_probs)
    return out


def _top_p_renorm_prob_fallback(
    probs: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = cumsum <= top_ps.view(-1, 1)
    # Always keep the first token.
    keep_mask[:, 0] = True
    sorted_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
    renorm = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sorted_probs = sorted_probs / renorm
    out = torch.zeros_like(probs)
    out.scatter_(1, sorted_idx, sorted_probs)
    return out


def _get_top_k_renorm_prob():
    if is_cuda() or is_hip():
        return top_k_renorm_prob
    return _top_k_renorm_prob_fallback


def _get_top_p_renorm_prob():
    if is_cuda() or is_hip():
        return top_p_renorm_prob
    return _top_p_renorm_prob_fallback


class SLRSWorker:
    """String-Level Rejection Sampling speculative worker (Algorithm 3 in SLRS.md).

    NOTE: This implementation currently uses one draft token proposal per step and
    rebuilds the draft context from the target string for correctness.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        if server_args.skip_tokenizer_init:
            raise ValueError("SLRS requires tokenizer initialization.")

        self.server_args = server_args
        self.target_worker = target_worker
        self.device = target_worker.device
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens or 1

        if self.speculative_num_draft_tokens != 1:
            logger.warning(
                "SLRS currently supports speculative_num_draft_tokens=1. "
                "Overriding to 1."
            )
            self.speculative_num_draft_tokens = 1

        # Initialize draft worker (separate tokenizer + KV cache pools).
        self.draft_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            pp_rank=0,  # FIXME
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
        )
        self.model_runner = self.draft_worker.model_runner
        self.model_config = self.draft_worker.model_config

        draft_tokenizer_path = (
            server_args.speculative_draft_model_path or server_args.model_path
        )
        draft_tokenizer_revision = (
            server_args.speculative_draft_model_revision or server_args.revision
        )
        self.draft_tokenizer = get_tokenizer(
            draft_tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=draft_tokenizer_revision,
        )

        self.target_vocab_size = target_worker.model_runner.model_config.vocab_size
        self.draft_vocab_size = self.draft_tokenizer.vocab_size

        self._draft_to_target = self._build_draft_to_target_map()
        self._draft_to_target_valid = (self._draft_to_target >= 0).nonzero(
            as_tuple=False
        ).view(-1)
        self._draft_to_target_valid_targets = self._draft_to_target[
            self._draft_to_target_valid
        ]
        self._draft_to_target_device = None
        self._draft_to_target_valid_device = None
        self._draft_to_target_valid_targets_device = None

        self.draft_tree_cache = RadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.draft_worker.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.draft_worker.model_runner.token_to_kv_pool_allocator,
                page_size=server_args.page_size,
                enable_kv_cache_events=bool(server_args.kv_events_config),
            )
        )

    def clear_cache_pool(self):
        # Draft cache is recreated per batch; just reset.
        self.draft_tree_cache.reset()

    def update_weights_from_tensor(self, recv_req):
        return self.draft_worker.update_weights_from_tensor(recv_req)

    def _build_draft_to_target_map(self) -> torch.Tensor:
        mapping = [-1] * self.draft_vocab_size
        target_tokenizer = self.target_worker.tokenizer
        for token_id in range(self.draft_vocab_size):
            text = self.draft_tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
            if not text:
                continue
            target_ids = target_tokenizer.encode(text, add_special_tokens=False)
            if not target_ids:
                continue
            t1 = target_ids[0]
            if 0 <= t1 < self.target_vocab_size:
                mapping[token_id] = t1
        return torch.tensor(mapping, dtype=torch.int64)

    def _get_draft_to_target_device(self, device: torch.device):
        if (
            self._draft_to_target_device is None
            or self._draft_to_target_device.device != device
        ):
            self._draft_to_target_device = self._draft_to_target.to(device)
            self._draft_to_target_valid_device = self._draft_to_target_valid.to(device)
            self._draft_to_target_valid_targets_device = (
                self._draft_to_target_valid_targets.to(device)
            )
        return (
            self._draft_to_target_device,
            self._draft_to_target_valid_device,
            self._draft_to_target_valid_targets_device,
        )

    def _clone_sampling_params(self, params: SamplingParams) -> SamplingParams:
        draft_params = copy.copy(params)
        draft_params.max_new_tokens = 1
        return draft_params

    def _build_draft_batch(self, batch: ScheduleBatch) -> ScheduleBatch:
        self.draft_tree_cache.reset()
        draft_reqs: List[Req] = []
        for req in batch.reqs:
            target_tokenizer = req.tokenizer
            full_text = target_tokenizer.decode(
                req.origin_input_ids + req.output_ids,
                skip_special_tokens=req.sampling_params.skip_special_tokens,
                spaces_between_special_tokens=req.sampling_params.spaces_between_special_tokens,
            )
            draft_input_ids = self.draft_tokenizer.encode(
                full_text, add_special_tokens=False
            )
            if not draft_input_ids and self.draft_tokenizer.bos_token_id is not None:
                draft_input_ids = [self.draft_tokenizer.bos_token_id]
            draft_req = Req(
                rid=f"{req.rid}-slrs",
                origin_input_text=full_text,
                origin_input_ids=draft_input_ids,
                sampling_params=self._clone_sampling_params(req.sampling_params),
                return_logprob=False,
                stream=False,
            )
            draft_req.tokenizer = self.draft_tokenizer
            draft_req.prefix_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            draft_req.init_next_round_input(self.draft_tree_cache)
            draft_reqs.append(draft_req)

        draft_batch = ScheduleBatch.init_new(
            reqs=draft_reqs,
            req_to_token_pool=self.draft_worker.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.draft_worker.model_runner.token_to_kv_pool_allocator,
            tree_cache=self.draft_tree_cache,
            model_config=self.draft_worker.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        draft_batch.prepare_for_extend()
        return draft_batch

    def _compute_probs(
        self, logits: torch.Tensor, sampling_info
    ) -> torch.Tensor:
        # Apply temperature
        probs = torch.softmax(logits / sampling_info.temperatures, dim=-1)
        top_k_fn = _get_top_k_renorm_prob()
        top_p_fn = _get_top_p_renorm_prob()
        if sampling_info.need_top_k_sampling:
            probs = top_k_fn(probs, sampling_info.top_ks)
        if sampling_info.need_top_p_sampling:
            probs = top_p_fn(probs, sampling_info.top_ps)
        if sampling_info.need_min_p_sampling:
            probs_before = probs
            max_probs = probs.max(dim=-1, keepdim=True).values
            thresholds = max_probs * sampling_info.min_ps.view(-1, 1)
            probs = torch.where(probs >= thresholds, probs, torch.zeros_like(probs))
            sums = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(sums > 0, probs / sums, probs_before)
        return probs

    def _compute_psi(
        self, draft_probs: torch.Tensor, target_vocab_size: int
    ) -> torch.Tensor:
        _, valid_idx, valid_targets = self._get_draft_to_target_device(
            draft_probs.device
        )
        draft_probs_valid = draft_probs[:, valid_idx]
        psi = torch.zeros(
            (draft_probs.shape[0], target_vocab_size),
            dtype=draft_probs.dtype,
            device=draft_probs.device,
        )
        psi.scatter_add_(
            1, valid_targets.view(1, -1).expand_as(draft_probs_valid), draft_probs_valid
        )
        return psi

    def _ensure_decode_prepared(self, batch: ScheduleBatch) -> None:
        if not batch.forward_mode.is_decode():
            return
        # For SLRS we want the normal decode preparation path.
        if batch.output_ids is None:
            return
        prev_algo = batch.spec_algorithm
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        batch.prepare_for_decode()
        batch.spec_algorithm = prev_algo

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            return self.target_worker.forward_batch_generation(model_worker_batch)

        self._ensure_decode_prepared(batch)

        # Target logits for p(t | context).
        model_worker_batch = batch.get_model_worker_batch()
        target_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output = target_result.logits_output
        sampling_info = batch.sampling_info

        # Apply custom logit processors and penalties on target logits.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits, sampling_info, num_tokens_in_batch=1
            )
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Draft logits for q(d | context) using a fresh prefill batch.
        draft_batch = self._build_draft_batch(batch)
        draft_worker_batch = draft_batch.get_model_worker_batch()
        draft_result = self.draft_worker.forward_batch_generation(
            draft_worker_batch, is_verify=True
        )
        draft_logits = draft_result.logits_output.next_token_logits

        # Free draft KV cache for this ephemeral batch.
        for req in draft_batch.reqs:
            release_kv_cache(req, self.draft_tree_cache, is_insert=False)

        # Compute proposal and target distributions.
        draft_probs = self._compute_probs(draft_logits, sampling_info)
        target_probs = self._compute_probs(
            logits_output.next_token_logits, sampling_info
        )
        psi = self._compute_psi(draft_probs, self.target_vocab_size)

        # Sample draft tokens.
        draft_sampled = torch.multinomial(draft_probs, num_samples=1).squeeze(1)
        draft_sampled_cpu = draft_sampled.tolist()

        accepted_token_ids: List[int] = []
        for i, req in enumerate(batch.reqs):
            text = self.draft_tokenizer.decode(
                [draft_sampled_cpu[i]],
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
            if not text:
                candidate = None
            else:
                target_ids = req.tokenizer.encode(text, add_special_tokens=False)
                candidate = target_ids[0] if target_ids else None

            if candidate is None or candidate >= self.target_vocab_size:
                # Fall back to target sampling.
                token_id = torch.multinomial(target_probs[i], num_samples=1).item()
                accepted_token_ids.append(token_id)
                continue

            p_val = target_probs[i, candidate].item()
            psi_val = psi[i, candidate].item()
            if psi_val <= 0 or p_val >= psi_val:
                accept = True
            else:
                accept = torch.rand((), device=target_probs.device).item() < (
                    p_val / psi_val
                )

            if accept:
                accepted_token_ids.append(candidate)
            else:
                residual = target_probs[i] - torch.minimum(target_probs[i], psi[i])
                residual_sum = residual.sum().item()
                if residual_sum <= 0:
                    token_id = torch.multinomial(target_probs[i], num_samples=1).item()
                else:
                    residual = residual / residual_sum
                    token_id = torch.multinomial(residual, num_samples=1).item()
                accepted_token_ids.append(token_id)

        if sampling_info.penalizer_orchestrator.is_required:
            sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(accepted_token_ids, dtype=torch.int64, device=self.device)
            )

        if batch.return_logprob:
            if SGLANG_RETURN_ORIGINAL_LOGPROB:
                logprobs = torch.nn.functional.log_softmax(
                    logits_output.next_token_logits, dim=-1
                )
            else:
                logprobs = torch.nn.functional.log_softmax(
                    logits_output.next_token_logits / sampling_info.temperatures, dim=-1
                )
            top_logprobs = None
            token_ids_logprobs = None
            if any(x > 0 for x in batch.top_logprobs_nums):
                top_logprobs = get_top_logprobs(logprobs, batch.top_logprobs_nums)
            if any(x is not None for x in batch.token_ids_logprobs):
                token_ids_logprobs = get_token_ids_logprobs(
                    logprobs, batch.token_ids_logprobs
                )

        for i, (req, token_id) in enumerate(zip(batch.reqs, accepted_token_ids)):
            req.output_ids.append(token_id)
            req.check_finished()
            if req.grammar is not None:
                req.grammar.accept_token(token_id)
                req.grammar.finished = req.finished()
            req.spec_verify_ct += 1

            if batch.return_logprob:
                req.output_token_logprobs_val.append(logprobs[i, token_id].item())
                req.output_token_logprobs_idx.append(token_id)
                if req.top_logprobs_num > 0 and top_logprobs is not None:
                    req.output_top_logprobs_val.append(top_logprobs[0][i])
                    req.output_top_logprobs_idx.append(top_logprobs[1][i])
                if req.token_ids_logprob is not None and token_ids_logprobs is not None:
                    req.output_token_ids_logprobs_val.append(token_ids_logprobs[0][i])
                    req.output_token_ids_logprobs_idx.append(token_ids_logprobs[1][i])

        next_token_ids = torch.tensor(
            accepted_token_ids, dtype=torch.int64, device=self.device
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[0] * len(accepted_token_ids),
            can_run_cuda_graph=target_result.can_run_cuda_graph,
        )
