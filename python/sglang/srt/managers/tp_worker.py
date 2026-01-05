# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Dict, List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    LoRAUpdateOutput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    DummyTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    is_npu,
    set_random_seed,
)
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils.patch_torch import register_sgl_tp_rank
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    initialize_dp_attention,
)

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


@dataclass
class NanoPearlQueuedRequest:
    rid: str
    prompt_ids: List[int]
    sampling_params: object
    is_stream: bool


@dataclass
class NanoPearlRequestState:
    seq_id: int
    done: bool
    is_stream: bool


class BaseTpWorker(ABC):
    @abstractmethod
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        pass

    @property
    @abstractmethod
    def model_runner(self) -> ModelRunner:
        pass

    @property
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa is not None

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return getattr(self.model_runner.attention_tp_group, "cpu_group", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self.model_runner.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):

        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        if getattr(self, "is_nano_pearl", False):
            return LoRAUpdateOutput(
                success=False,
                error_message="nano-pearl mode does not support LoRA adapters.",
            )
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        if getattr(self, "is_nano_pearl", False):
            return LoRAUpdateOutput(
                success=False,
                error_message="nano-pearl mode does not support LoRA adapters.",
            )
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    def can_run_lora_batch(self, lora_ids: list[str]) -> bool:
        if getattr(self, "is_nano_pearl", False):
            return False
        lora_ids_set = set(lora_ids) if isinstance(lora_ids, list) else lora_ids
        return self.model_runner.lora_manager.validate_lora_batch(lora_ids_set)

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch).logits_output
        embeddings = logits_output.embeddings
        return embeddings


class NanoPearlStubRunner:
    """A lightweight model runner that avoids loading target weights in nano-pearl mode."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        max_total_num_tokens: int,
        max_num_seqs: int,
    ):
        self.model_config = model_config
        self.server_args = server_args
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.page_size = server_args.page_size
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = model_config.is_hybrid_swa_compress
        self.attention_chunk_size = model_config.attention_chunk_size
        self.dtype = model_config.dtype
        self.kv_cache_memory = 0
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None
        self.model = None
        self.sampler = None
        self.sliding_window_size = None

        set_global_server_args_for_scheduler(server_args)
        self._init_distributed(nccl_port)

        if self.page_size != 1:
            raise RuntimeError("nano-pearl stub runner requires page_size=1.")

        self.max_total_num_tokens = self._normalize_max_tokens(max_total_num_tokens)
        self.full_max_total_num_tokens = self.max_total_num_tokens
        self.swa_max_total_num_tokens = self.max_total_num_tokens
        self.max_running_requests = max_num_seqs

        extra_max_context_len = 4
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_seqs,
            max_context_len=model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=server_args.enable_memory_saver,
        )

        need_sort = server_args.disaggregation_mode in ("decode", "prefill")
        self.token_to_kv_pool_allocator = DummyTokenToKVPoolAllocator(
            size=self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            need_sort=need_sort,
        )
        self.token_to_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()

    def _normalize_max_tokens(self, max_total_num_tokens: int) -> int:
        if max_total_num_tokens <= 0:
            raise RuntimeError("nano-pearl requires max_total_num_tokens > 0.")
        return max_total_num_tokens // self.page_size * self.page_size

    def _init_distributed(self, nccl_port: int) -> None:
        torch.get_device_module(self.device).set_device(self.gpu_id)
        if self.device == "cuda":
            backend = "nccl"
        elif self.device == "xpu":
            backend = "xccl"
        elif self.device == "hpu":
            backend = "hccl"
        elif self.device == "npu":
            backend = "hccl"
        else:
            backend = "gloo"

        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{nccl_port}"

        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        init_distributed_environment(
            backend=backend,
            world_size=self.tp_size * self.pp_size,
            rank=self.tp_size * self.pp_rank + self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=dist_init_method,
            timeout=self.server_args.dist_timeout,
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=self.tp_size,
            expert_model_parallel_size=self.moe_ep_size,
            pipeline_model_parallel_size=self.pp_size,
        )
        initialize_dp_attention(
            server_args=self.server_args,
            model_config=self.model_config,
        )
        if is_npu():
            register_sgl_tp_rank(self.gpu_id)

        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

    @property
    def max_token_pool_size(self):
        return (
            min(self.swa_max_total_num_tokens, self.max_total_num_tokens)
            if self.is_hybrid_swa
            else self.max_total_num_tokens
        )

    @property
    def hybrid_gdn_config(self):
        return None

    @property
    def mamba2_config(self):
        return None

    @property
    def kimi_linear_config(self):
        return None

    @property
    def mambaish_config(self):
        return None

    def update_weights_from_disk(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def init_weights_update_group(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def destroy_weights_update_group(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def init_weights_send_group_for_remote_instance(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def send_weights_to_remote_instance(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def update_weights_from_distributed(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def update_weights_from_tensor(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def update_weights_from_ipc(self, *args, **kwargs):
        return False, "nano-pearl stub runner does not support weight updates."

    def get_weights_by_name(self, *args, **kwargs):
        raise RuntimeError("nano-pearl stub runner does not support weight queries.")

    def load_lora_adapter(self, *args, **kwargs):
        raise RuntimeError("nano-pearl stub runner does not support LoRA.")

    def unload_lora_adapter(self, *args, **kwargs):
        raise RuntimeError("nano-pearl stub runner does not support LoRA.")

    def forward(self, *args, **kwargs):
        raise RuntimeError("nano-pearl stub runner does not execute forward passes.")


class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank

        # MTP model runners
        self.model_runner_list = []

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            model_revision=(
                server_args.revision
                if not is_draft_worker
                else server_args.speculative_draft_model_revision
            ),
            is_draft_model=is_draft_worker,
        )

        # Init DLLM algorithm
        if server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(server_args)
        else:
            self.dllm_algorithm = None

        self.is_nano_pearl = server_args.enable_nano_pearl
        self.use_pearl_engine = self.is_nano_pearl
        self.pearl_engine = None
        self._nano_pearl_sampling_cls = None
        self._nano_pearl_pending_tokens: Dict[str, deque] = {}
        self._nano_pearl_seq_id_to_rid: Dict[int, str] = {}
        self._nano_pearl_generated: set[str] = set()
        self._nano_pearl_active: Dict[str, NanoPearlRequestState] = {}
        self._nano_pearl_warned_sampling = False
        self._nano_pearl_logged_sampling = False
        self._nano_pearl_lock = threading.Lock()
        self._nano_pearl_cv = threading.Condition()
        self._nano_pearl_request_queue: Deque[NanoPearlQueuedRequest] = deque()
        self._nano_pearl_worker_thread: Optional[threading.Thread] = None
        self._nano_pearl_worker_shutdown = False
        self._nano_pearl_stream_error: Optional[BaseException] = None
        self._nano_pearl_max_num_batched_tokens: Optional[int] = None
        self._nano_pearl_max_num_seqs: Optional[int] = None
        self._nano_pearl_wait_timeout_s = float(
            os.getenv("NANO_PEARL_SGLANG_WAIT_TIMEOUT_S", "5")
        )
        self._nano_pearl_stream_wait_timeout_s = float(
            os.getenv("NANO_PEARL_SGLANG_STREAM_WAIT_TIMEOUT_S", "1")
        )
        self._nano_pearl_last_wait_warn_ts = 0.0
        self._nano_pearl_prefetch_steps = max(
            int(os.getenv("NANO_PEARL_SGLANG_PREFETCH_STEPS", "2")), 1
        )

        if self.use_pearl_engine:
            self._init_pearl_engine(server_args)
            self._start_nano_pearl_worker()

        if self.use_pearl_engine:
            max_total_num_tokens = (
                self._nano_pearl_max_num_batched_tokens
                or server_args.max_total_tokens
                or server_args.max_prefill_tokens
                or self.model_config.context_len
            )
            max_num_seqs = (
                self._nano_pearl_max_num_seqs
                or server_args.max_running_requests
                or 512
            )
            self._model_runner = NanoPearlStubRunner(
                model_config=self.model_config,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                tp_size=server_args.tp_size,
                moe_ep_rank=moe_ep_rank,
                moe_ep_size=server_args.ep_size,
                pp_rank=pp_rank,
                pp_size=server_args.pp_size,
                nccl_port=nccl_port,
                max_total_num_tokens=max_total_num_tokens,
                max_num_seqs=max_num_seqs,
            )
        else:
            self._model_runner = ModelRunner(
                model_config=self.model_config,
                mem_fraction_static=server_args.mem_fraction_static,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                tp_size=server_args.tp_size,
                moe_ep_rank=moe_ep_rank,
                moe_ep_size=server_args.ep_size,
                pp_rank=pp_rank,
                pp_size=server_args.pp_size,
                nccl_port=nccl_port,
                dp_rank=dp_rank,
                server_args=server_args,
                is_draft_worker=is_draft_worker,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                draft_model_idx=0 if is_multi_layer_eagle else None,
            )
        if is_multi_layer_eagle:
            self.model_runner_list.append(self.model_runner)
            for i in range(1, server_args.speculative_num_steps):
                self.model_runner_list.append(
                    ModelRunner(
                        model_config=self.model_config,
                        mem_fraction_static=server_args.mem_fraction_static,
                        gpu_id=gpu_id,
                        tp_rank=tp_rank,
                        tp_size=server_args.tp_size,
                        moe_ep_rank=moe_ep_rank,
                        moe_ep_size=server_args.ep_size,
                        pp_rank=pp_rank,
                        pp_size=server_args.pp_size,
                        nccl_port=nccl_port,
                        dp_rank=dp_rank,
                        server_args=server_args,
                        is_draft_worker=is_draft_worker,
                        req_to_token_pool=req_to_token_pool,
                        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                        draft_model_idx=i,
                    )
                )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = self.model_runner.max_running_requests
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = (
            server_args.speculative_algorithm is not None and not self.is_nano_pearl
        )
        self.hicache_layer_transfer_counter = None

    @property
    def model_runner(self) -> ModelRunner:
        return self._model_runner

    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def is_dllm(self):
        return self.dllm_algorithm is not None

    def _forward_batch_generation_dllm(
        self, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        logits_output, next_token_ids, can_run_cuda_graph = self.dllm_algorithm.run(
            self.model_runner, forward_batch
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_batch_generation_nano_pearl(
        self,
        model_worker_batch: ModelWorkerBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        skip_attn_backend_init: bool = False,
    ) -> GenerationBatchResult:
        """Run nano-pearl engine to generate tokens and feed them back to the scheduler."""
        if self.pearl_engine is None:
            raise RuntimeError("nano-pearl engine is not initialized.")

        if model_worker_batch.return_logprob:
            raise RuntimeError("nano-pearl engine does not support logprob outputs.")

        if model_worker_batch.is_prefill_only:
            raise RuntimeError("nano-pearl engine does not support prefill-only batches.")

        new_reqs = [
            req for req in model_worker_batch.reqs
            if req.rid not in self._nano_pearl_generated
        ]
        if new_reqs:
            self._nano_pearl_enqueue_reqs(new_reqs)

        self._nano_pearl_wait_for_tokens(model_worker_batch.reqs)

        is_prefill = model_worker_batch.forward_mode.is_extend()
        next_token_ids: List[int] = []
        nano_pearl_output_ids: List[List[int]] = []
        with self._nano_pearl_cv:
            stream_error = self._nano_pearl_stream_error
            no_engine_active = (
                not self._nano_pearl_request_queue
                and not self._nano_pearl_seq_id_to_rid
                and not self._nano_pearl_active
            )
            for req in model_worker_batch.reqs:
                token_queue = self._nano_pearl_pending_tokens.get(req.rid)
                state = self._nano_pearl_active.get(req.rid)

                if stream_error is not None:
                    token_id = self._nano_pearl_fallback_token(req)
                    next_token_ids.append(token_id)
                    nano_pearl_output_ids.append([])
                    if state is not None and state.done:
                        self._nano_pearl_active.pop(req.rid, None)
                    continue
                if not token_queue:
                    if state is None or not state.done:
                        if no_engine_active:
                            token_id = self._nano_pearl_fallback_token(req)
                            next_token_ids.append(token_id)
                            nano_pearl_output_ids.append([])
                        else:
                            # -1 means no token yet; scheduler should skip update.
                            next_token_ids.append(-1)
                            nano_pearl_output_ids.append([])
                        continue
                    token_id = self._nano_pearl_fallback_token(req)
                    next_token_ids.append(token_id)
                    nano_pearl_output_ids.append([])
                    if state is not None and state.done:
                        self._nano_pearl_active.pop(req.rid, None)
                    continue

                if req.stream:
                    token_id = token_queue.popleft()
                    next_token_ids.append(token_id)
                    nano_pearl_output_ids.append([])
                else:
                    token_ids = list(token_queue)
                    token_queue.clear()
                    if not token_ids:
                        token_ids = [self._nano_pearl_fallback_token(req)]
                    next_token_ids.append(token_ids[0])
                    nano_pearl_output_ids.append(token_ids)

                if token_queue is not None and not token_queue:
                    self._nano_pearl_pending_tokens.pop(req.rid, None)
                    if state is not None and state.done:
                        self._nano_pearl_active.pop(req.rid, None)

        next_token_device = torch.device("cpu")
        if not self.server_args.disable_overlap_schedule:
            if model_worker_batch.input_ids is not None:
                next_token_device = model_worker_batch.input_ids.device
            else:
                next_token_device = torch.device(self.device)
        next_token_ids_tensor = torch.tensor(
            next_token_ids, dtype=torch.long, device=next_token_device
        )
        logits_output = LogitsProcessorOutput(next_token_logits=None)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids_tensor,
            can_run_cuda_graph=False,
            nano_pearl_output_ids=nano_pearl_output_ids,
        )

    def _ensure_nano_pearl_importable(self):
        nano_pearl_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "nano-PEARL")
        )
        if nano_pearl_root not in sys.path:
            sys.path.append(nano_pearl_root)

    def _init_pearl_engine(self, server_args: ServerArgs):
        self._ensure_nano_pearl_importable()
        from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams

        if server_args.tp_size != 1 or server_args.pp_size != 1:
            raise RuntimeError(
                "nano-pearl engine requires sglang tp_size=1 and pp_size=1. "
                "Use --nano-pearl-target-tp-size to set PEARL TP."
            )
        target_tp_size = (
            server_args.nano_pearl_target_tp_size or server_args.tp_size
        )
        required_gpus = server_args.draft_model_tp_size + target_tp_size
        available_gpus = torch.cuda.device_count()
        if available_gpus < required_gpus:
            raise RuntimeError(
                "nano-pearl requires at least %d GPUs (draft tp size + target tp size), "
                "but only %d CUDA device(s) are available."
                % (required_gpus, available_gpus)
            )

        max_num_batched_tokens = (
            server_args.max_total_tokens
            or server_args.max_prefill_tokens
            or 16384
        )
        max_num_seqs = server_args.max_running_requests or 512
        if server_args.max_running_requests is not None:
            # Allow a small headroom for chunked-prefill overlap to avoid bs > max_num_seqs.
            max_num_seqs += 1
        max_model_len = self.model_config.context_len

        if max_num_batched_tokens < max_model_len:
            logger.warning(
                "nano-pearl requires max_num_batched_tokens >= max_model_len; "
                "raising max_num_batched_tokens from %s to %s.",
                max_num_batched_tokens,
                max_model_len,
            )
            max_num_batched_tokens = max_model_len

        self._nano_pearl_max_num_batched_tokens = max_num_batched_tokens
        self._nano_pearl_max_num_seqs = max_num_seqs
        gamma_env = os.getenv("NANO_PEARL_GAMMA")
        if gamma_env:
            try:
                gamma = int(gamma_env)
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid NANO_PEARL_GAMMA value: {gamma_env}"
                ) from exc
        else:
            gamma = -1
        config = PEARLConfig(
            server_args.speculative_draft_model_path,
            server_args.model_path,
            draft_tensor_parallel_size=server_args.draft_model_tp_size,
            target_tensor_parallel_size=target_tp_size,
            share_draft_target_gpus=False,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=(
                server_args.mem_fraction_static
                if server_args.mem_fraction_static is not None
                else 0.9
            ),
            gamma=gamma,
        )
        self.pearl_engine = PEARLEngine(config)
        self._nano_pearl_sampling_cls = SamplingParams

    def _nano_pearl_sampling_params(self, sampling_params, max_new_tokens=None):
        if (
            not self._nano_pearl_warned_sampling
            and (
                sampling_params.top_p != 1.0
                or sampling_params.top_k not in (None, -1, 1 << 30)
            )
        ):
            logger.warning(
                "nano-pearl only supports temperature/max_tokens/ignore_eos. "
                "top_p/top_k will be ignored."
            )
            self._nano_pearl_warned_sampling = True
        if max_new_tokens is None:
            max_new_tokens = sampling_params.max_new_tokens
        if not self._nano_pearl_logged_sampling:
            self._nano_pearl_logged_sampling = True
        return self._nano_pearl_sampling_cls(
            temperature=sampling_params.temperature,
            max_tokens=max_new_tokens,
            ignore_eos=sampling_params.ignore_eos,
        )

    def _start_nano_pearl_worker(self):
        if self._nano_pearl_worker_thread is not None:
            return

        def _worker():
            while True:
                try:
                    while True:
                        with self._nano_pearl_cv:
                            while (
                                not self._nano_pearl_request_queue
                                and not self._nano_pearl_seq_id_to_rid
                                and not self._nano_pearl_worker_shutdown
                            ):
                                self._nano_pearl_cv.wait()
                            if self._nano_pearl_worker_shutdown:
                                return
                            batch = list(self._nano_pearl_request_queue)
                            self._nano_pearl_request_queue.clear()
                            if batch:
                                self._nano_pearl_stream_error = None

                        if self.pearl_engine is None:
                            with self._nano_pearl_cv:
                                self._nano_pearl_stream_error = RuntimeError(
                                    "nano-pearl engine is not initialized."
                                )
                                for rid in list(self._nano_pearl_active):
                                    state = self._nano_pearl_active.get(rid)
                                    if state is not None:
                                        state.done = True
                                self._nano_pearl_seq_id_to_rid.clear()
                                self._nano_pearl_cv.notify_all()
                            continue

                        if batch:
                            with self._nano_pearl_lock:
                                for req in batch:
                                    seq_id = self.pearl_engine.add_request(
                                        req.prompt_ids, req.sampling_params
                                    )
                                    with self._nano_pearl_cv:
                                        self._nano_pearl_seq_id_to_rid[seq_id] = (
                                            req.rid
                                        )
                                        self._nano_pearl_active[req.rid] = (
                                            NanoPearlRequestState(
                                                seq_id=seq_id,
                                                done=False,
                                                is_stream=req.is_stream,
                                            )
                                        )
                                        self._nano_pearl_pending_tokens.setdefault(
                                            req.rid, deque()
                                        )

                        with self._nano_pearl_cv:
                            has_active = bool(self._nano_pearl_seq_id_to_rid)
                            has_pending = bool(self._nano_pearl_request_queue)
                        if not has_active and not has_pending:
                            break

                        done = False
                        for _ in range(self._nano_pearl_prefetch_steps):
                            with self._nano_pearl_lock:
                                step_output, step_done = (
                                    self.pearl_engine.stream_generate_step()
                                )
                            with self._nano_pearl_cv:
                                for seq_id, token_ids in step_output:
                                    rid = self._nano_pearl_seq_id_to_rid.get(seq_id)
                                    if rid is None:
                                        continue
                                    self._nano_pearl_pending_tokens.setdefault(
                                        rid, deque()
                                    ).extend(token_ids)
                                if step_done:
                                    for rid in list(self._nano_pearl_active):
                                        state = self._nano_pearl_active.get(rid)
                                        if state is not None:
                                            state.done = True
                                    self._nano_pearl_seq_id_to_rid.clear()
                                self._nano_pearl_cv.notify_all()
                                if step_done or self._nano_pearl_request_queue:
                                    done = step_done
                                    break
                        if done:
                            continue
                except Exception as exc:
                    logger.error("nano-pearl: stream_generate_step failed: %s", exc)
                    with self._nano_pearl_cv:
                        self._nano_pearl_stream_error = exc
                        for rid in list(self._nano_pearl_active):
                            state = self._nano_pearl_active.get(rid)
                            if state is not None:
                                state.done = True
                        self._nano_pearl_seq_id_to_rid.clear()
                        self._nano_pearl_cv.notify_all()

        self._nano_pearl_worker_thread = threading.Thread(
            target=_worker, daemon=True
        )
        self._nano_pearl_worker_thread.start()

    def _nano_pearl_enqueue_reqs(self, reqs):
        queued = []
        for req in reqs:
            if req.stream:
                req.nano_pearl_chunked = True
            prompt_ids = self._nano_pearl_get_prompt_ids(req)
            max_new_tokens = req.sampling_params.max_new_tokens
            max_new_tokens_limit = max(self.max_req_len - len(prompt_ids) - 1, 1)
            if max_new_tokens is None:
                max_new_tokens = max_new_tokens_limit
            else:
                max_new_tokens = min(max_new_tokens, max_new_tokens_limit)
            nano_sampling = self._nano_pearl_sampling_params(
                req.sampling_params, max_new_tokens=max_new_tokens
            )
            queued.append(
                NanoPearlQueuedRequest(
                    rid=req.rid,
                    prompt_ids=prompt_ids,
                    sampling_params=nano_sampling,
                    is_stream=req.stream,
                )
            )

        with self._nano_pearl_cv:
            for item in queued:
                self._nano_pearl_request_queue.append(item)
                self._nano_pearl_generated.add(item.rid)
                self._nano_pearl_pending_tokens.setdefault(item.rid, deque())
            self._nano_pearl_cv.notify_all()

    def _nano_pearl_wait_for_tokens(self, reqs):
        timeout = self._nano_pearl_wait_timeout_s
        if any(req.stream for req in reqs):
            timeout = min(timeout, self._nano_pearl_stream_wait_timeout_s)
        require_all_ready = not self.server_args.disable_overlap_schedule
        deadline = time.monotonic() + timeout
        with self._nano_pearl_cv:
            while True:
                if self._nano_pearl_stream_error is not None:
                    return
                has_pending = bool(self._nano_pearl_request_queue)
                has_active = bool(self._nano_pearl_seq_id_to_rid)
                if not has_pending and not has_active:
                    all_done = True
                    for state in self._nano_pearl_active.values():
                        if state is not None and not state.done:
                            all_done = False
                            break
                    if all_done:
                        return
                if require_all_ready:
                    all_ready = True
                    for req in reqs:
                        token_queue = self._nano_pearl_pending_tokens.get(req.rid)
                        state = self._nano_pearl_active.get(req.rid)
                        if token_queue:
                            continue
                        if state is None or not state.done:
                            all_ready = False
                            break
                    if all_ready:
                        return
                    if timeout > 0 and time.monotonic() >= deadline:
                        now = time.monotonic()
                        if now - self._nano_pearl_last_wait_warn_ts > 5:
                            self._nano_pearl_last_wait_warn_ts = now
                            logger.warning(
                                "nano-pearl wait timeout (%.2fs). pending=%d active=%d",
                                timeout,
                                len(self._nano_pearl_request_queue),
                                len(self._nano_pearl_seq_id_to_rid),
                            )
                        deadline = time.monotonic() + timeout
                    self._nano_pearl_cv.wait(timeout=0.05)
                    continue

                any_ready = False
                for req in reqs:
                    token_queue = self._nano_pearl_pending_tokens.get(req.rid)
                    state = self._nano_pearl_active.get(req.rid)
                    if token_queue:
                        any_ready = True
                        continue
                    if state is None:
                        continue
                    if state.done:
                        any_ready = True
                        continue
                if any_ready:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if timeout > 0:
                        now = time.monotonic()
                        if now - self._nano_pearl_last_wait_warn_ts > 5:
                            self._nano_pearl_last_wait_warn_ts = now
                            logger.warning(
                                "nano-pearl wait timeout (%.2fs). pending=%d active=%d",
                                timeout,
                                len(self._nano_pearl_request_queue),
                                len(self._nano_pearl_seq_id_to_rid),
                            )
                    return
                self._nano_pearl_cv.wait(timeout=min(0.05, remaining))

    def _nano_pearl_get_prompt_ids(self, req):
        if req.origin_input_ids:
            return req.origin_input_ids
        if req.origin_input_text:
            return self.pearl_engine.tokenizer.encode(
                req.origin_input_text,
                add_special_tokens=False,
            )
        return []

    def _nano_pearl_fallback_token(self, req):
        if req.eos_token_ids:
            return next(iter(req.eos_token_ids))
        eos_ids = self.model_config.hf_eos_token_id or set()
        if eos_ids:
            return next(iter(eos_ids))
        return 0

    def get_remote_instance_transfer_engine_info(self):
        return (
            self.model_runner.remote_instance_transfer_engine_session_id,
            self.model_runner.remote_instance_transfer_engine_weight_info,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        # FIXME(lsyin): maybe remove skip_attn_backend_init in forward_batch_generation,
        #               which requires preparing replay to always be in this function

        # Get forward batch from model worker batch
        if model_worker_batch is not None:
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)

            if self.is_nano_pearl:
                return self._forward_batch_generation_nano_pearl(
                    model_worker_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                    skip_attn_backend_init=skip_attn_backend_init,
                )

            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None
            if self.is_nano_pearl:
                raise RuntimeError(
                    "nano-pearl forward path requires model_worker_batch input."
                )

        if self.is_dllm():
            return self._forward_batch_generation_dllm(forward_batch)

        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

            if is_verify:
                # Skip sampling and return logits for target forward
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and model_worker_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not model_worker_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(model_worker_batch.seq_lens),
                    dtype=torch.long,
                    device=model_worker_batch.input_ids.device,
                )
                if (
                    model_worker_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, model_worker_batch
                    )

            return batch_result
        else:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            batch.split_forward_batch = forward_batch
            batch.seq_lens_cpu_cache = model_worker_batch.seq_lens_cpu
        else:
            model_worker_batch = batch.get_model_worker_batch(batch.seq_lens_cpu_cache)

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
