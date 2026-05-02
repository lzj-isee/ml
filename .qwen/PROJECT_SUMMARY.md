The user wants me to generate a project summary based on the conversation history. Let me analyze the key points:

1. **Overall Goal**: The user is preparing for interviews for LLM algorithm engineer positions, focusing on implementing ML algorithms using NumPy/Torch.

2. **Key Knowledge**:
- Working directory: `/home/lzj/codes/ml`
- Conda environment: `py312` (activated via `/home/lzj/miniforge3`)
- User prefers PyTorch over NumPy (but wants to practice NumPy implementations)
- Testing approach: compare NumPy implementations with PyTorch for verification
- User has deep learning background, uses PyTorch regularly

3. **Recent Actions**:
- Created learning roadmap document at `docs/learning_roadmap.md`
- Created and refined interview Q&A document at `docs/llm_interview_qa.md`
- Implemented Linear layer with forward/backward in `implementations/linear.py` with tests
- Started implementing MHA/GQA in `implementations/mha.py`
- Learned about einsum, attention mask, causal mask, etc.
- Committed changes but push was cancelled

4. **Current Plan**:
- User was practicing MHA implementation
- Tomorrow plans to write FFN and RoPE
- The commit was done but push was cancelled by user

Let me format this properly.# Project Summary

## Overall Goal
User is preparing for LLM algorithm engineer interviews, practicing hand-writing ML algorithms (MLP, Conv, Attention, Norm layers, MoE, etc.) using PyTorch/NumPy, with PyTorch verification for correctness.

## Key Knowledge
- **Project Directory**: `/home/lzj/codes/ml`
- **Conda Environment**: `py312` activated via `source /home/lzj/miniforge3/etc/profile.d/conda.sh`
- **Testing Convention**: Compare NumPy implementations with PyTorch `nn.Module` equivalents using `np.allclose()`/`torch.allclose()`
- **User Background**: Deep learning experience, uses PyTorch regularly, new to NumPy implementations
- **Linear Layer Weight Shape**: `(d_out, d_in)` - matches PyTorch convention
- **Backward Gradients**: Sum over batch (not mean), consistent with loss aggregation
- **einsum Rule**: Dimensions not in output are summed automatically

## Recent Actions
- Created `docs/learning_roadmap.md` with structured interview prep plan (stages 1-8 covering activations, norms, attention, transformer, MoE, optimizers)
- Created `docs/llm_interview_qa.md` from imported notes, cleaned HTML tags, organized by topic, marked unanswered questions as TODO
- Implemented `implementations/linear.py` with `linear_forward` and `linear_backward`, added torch verification tests (all passed)
- Started `implementations/mha.py`:
  - `RMSNorm` class (mean-based variance, float32 computation with dtype restore)
  - `Attention` class with GQA support (`num_q`, `num_kv` heads)
  - Q/K/V projections, q_norm/k_norm (Qwen3-style)
  - `expand_as` for KV repetition before transpose
  - Causal mask using `torch.tril` + `torch.where`
- Discussed mask implementation: `torch.finfo(dtype).min` vs `float('-inf')` (finfo more universal for float16)
- Commit created but **push cancelled by user**

## Current Plan
1. [DONE] Linear layer implementation + tests
2. [IN PROGRESS] MHA/GQA implementation (basic attention working, mask correct)
3. [TODO] Add FFN (SwiGLU style) to `mha.py`
4. [TODO] Add RoPE (Rotary Position Embedding)
5. [TODO] Continue other modules per roadmap
6. [TODO] Push commit when ready (user cancelled during this session)

## File Structure
```
ml/
├── docs/
│   ├── learning_roadmap.md    # Interview prep roadmap
│   ├── llm_interview_qa.md    # Q&A notes (organized, TODOs marked)
│   └── _llm_qa.md             # Original raw notes (preserved)
├── implementations/
│   ├── linear.py              # Linear + tests (✓ passed)
│   └── mha.py                 # RMSNorm + Attention (GQA) in progress
```

## Pending Commit
```
feat: add GQA/MHA implementation and update interview doc
- Add Attention class with GQA support (Qwen3-style)
- Add RMSNorm implementation
- Update llm_interview_qa.md with new questions
```
Commit hash: `a98c497` - ready to push when user confirms.

---

## Summary Metadata
**Update time**: 2026-04-11T17:56:09.687Z 
