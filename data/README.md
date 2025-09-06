GPT-2 Generative QA Baseline (SQuAD 10%)
Task: Generative QA with GPT-2

question: {q}
context: {ctx}
answer:


Dataset: SQuAD v1.1 → 10% train / 10% val

Eval (Day 1): approx EM / F1 (case+punct normalized) + PPL

Logging: TensorBoard (or wandb) + meta.json

Outputs:

outputs/ckpt/ → checkpoints

outputs/samples/ → generated answers

outputs/logs/ → metrics

outputs/meta.json → run info

Notes: use EOS as pad token; reduce max_len if OOM