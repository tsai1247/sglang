# SLRS (String-Level Rejection Sampling, Algorithm 3)
## ref
- name: Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies
- url: https://openreview.net/pdf?id=vQubr1uBUw
## 基本概念
- Drafter 先根據自己的詞表生成一段 token 序列，並 decode 成字串 
s。  
 
- Target 把同一個 prefix 丟進去，直接在字串空間上檢驗「如果由 Target 來產生，會不會也產生這個字串  s ？」
​ 

- 核心機制
    - SLRS 在字串層級做「拒絕採樣 (rejection sampling)」：
        視 Drafter 產出的字串 s 為候選樣本。 
        Target 用自己的條件機率 p_T(s | context )  去檢查這個字串是否可以被接受，並用接受率保證最終樣本分佈仍然是 Target 的真實分佈。
​        因為比對的是完整字串，而不是逐 token 的對齊，因此可以完全繞開兩個 tokenizer 的切詞差異，只要 decode 出來的文字一樣就行。


- pseudo code
```
Algorithm: SLRS (String-Level Rejection Sampling Verification)
Input:
  - p: probability distribution over target vocabulary T
  - q: probability distribution over drafter vocabulary D
  - T(·): mapping from D* to T* (string-level tokenization from drafter to target)
  - S1: lookahead indicator function, takes current length i and returns a boolean

Output:
  - One token t from target vocabulary T

Procedure:
  1. Sample drafter tokens d1, d2, ... sequentially from q
     until the index i satisfies S1(i).

  2. Let s = d1 ⊕ d2 ⊕ ... ⊕ di be the concatenated drafter string.

  3. Apply the target tokenizer:
       (t1, t2, ..., tm) ← T(s)

  4. Compute the auxiliary distribution ψ over T implied by q and T(·).
     (ψ(t1) is the probability mass from q that maps to t1.)

  5. If p(t1) ≥ ψ(t1), then:
       - Accept t1 and return t1.

  6. Otherwise:
       - With probability p(t1) / ψ(t1), accept t1 and return t1.

  7. If t1 is rejected:
       - Define a residual distribution over T:
           r(t) ∝ p(t) − min{ p(t), ψ(t) }
         i.e.
           r(t) = ( p(t) − min{ p(t), ψ(t) } )
                  / ( 1 − Σ_{t'} min{ p(t'), ψ(t') } )

       - Sample t from r(t) and return t.
```