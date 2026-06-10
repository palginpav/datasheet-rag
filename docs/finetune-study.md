# Fine-tuning study: does QLoRA help datasheet QA?

**Question.** RAG supplies knowledge at query time; fine-tuning bakes it into weights.
For exact-value datasheet QA, which wins — and do they compose?

**Setup.** QLoRA on Qwen3-4B (4-bit NF4 base, LoRA r=16 on attention + MLP, 33M trainable
params / 0.81%), trained on 243 synthetic question→answer pairs generated from the corpus
(3 epochs; eval loss 7.5 → 2.11, token accuracy 0.53 → 0.81). Evaluated over the 100-question
golden set. To keep the RAG arms strictly comparable, dense+rerank contexts were **frozen
once** and both models answered from the identical contexts — the model is the only variable.
Correctness is judged 1–5 by gpt-oss; refusal is measured on the 9 unanswerable questions.

## Results

| Arm | correctness (1–5) | must-include | refusals (of 9) |
|---|---|---|---|
| base, closed-book | 2.25 | 0.45 | 8/9 |
| base + RAG | **4.62** | **0.87** | 8/9 |
| FT, closed-book | 1.95 | 0.37 | **3/9** |
| FT + RAG | 4.56 | 0.79 | 9/9 |

## Findings

1. **Retrieval dominates; knowledge does not fit in the weights.** Both closed-book arms
   (base 2.25, FT 1.95) are far below their RAG counterparts (~4.6). Exact spec values —
   "538 µA", "2.048 V", "STM32C031C4 → 32 KB" — are precisely what a 4B model cannot
   memorize and what retrieval supplies perfectly. For this task, RAG is not optional.

2. **QLoRA did not inject useful knowledge — it made closed-book *worse*.** FT closed-book
   (1.95) underperforms base closed-book (2.25). 243 examples cannot encode thousands of
   distinct datasheet values; what the model learned instead was the *style* of always
   producing a confident short answer.

3. **The cost of that style: refusal calibration collapsed.** FT closed-book correctly
   refused only **3/9** unanswerable questions vs the base model's 8/9 — because the
   synthetic training set contained no "I don't know" examples, fine-tuning taught the model
   to always answer, i.e. to hallucinate confidently. This is the clearest result in the
   study and a standard fine-tuning failure mode: training on answers-only erodes the
   model's ability to abstain.

4. **FT + RAG ≈ base + RAG (slightly worse).** With retrieval present, fine-tuning neither
   helped nor clearly hurt correctness (4.56 vs 4.62), and must-include coverage dropped
   (0.79 vs 0.87). The one upside — FT+RAG refused 9/9 — is within noise on nine questions.

## Conclusion

**For exact-value datasheet QA on a corpus this size, retrieval is the entire game, and
QLoRA on synthetic Q/A is not worth it** — it fails to add knowledge, degrades abstention,
and does not improve the RAG pipeline. Fine-tuning would only pay off for *behavioral*
adaptation (output format, domain tone, a fixed answer schema) with refusal examples
included in the training mix — not for injecting factual specs, which is retrieval's job.

The adapter, training config, and per-arm traces are reproducible from `scripts/train_qlora.py`
and `scripts/eval_ft.py`; the trained adapter is small (66 MB) but intentionally not the
shipped path — the project ships **dense + rerank**, the configuration the evidence supports.
