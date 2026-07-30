# GPU box setup

Target: one 24 GB card (RTX 4090 / A10 / L4) on RunPod, Lambda, Modal, or Vast. A Stage-A LoRA run
on 1–3B is minutes to low hours. Rent by the hour; there's no reason to buy hardware for this.

## Once, on a fresh box

```bash
# Pick an image with CUDA 12.x + PyTorch preinstalled (RunPod's "PyTorch 2.4" template is fine).
git clone <your-repo> writelikeme && cd writelikeme
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu]"
nvidia-smi                     # confirm the card and driver
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the base model is gated on Hugging Face, `export HF_TOKEN=...` and `huggingface-cli login`.
Qwen2.5 is not gated.

## Move the data over

The laptop produces everything the GPU needs. Nothing on the GPU box needs your raw corpus — only
the scrubbed, split pairs and the fitted verifier.

```bash
# from the laptop
rsync -av data/processed/ GPUBOX:~/writelikeme/data/processed/
rsync -av runs/av/        GPUBOX:~/writelikeme/runs/av/
```

Then pull results back:

```bash
rsync -av GPUBOX:~/writelikeme/runs/ runs/
```

Adapters are tens of MB, so this is fast. Keep the raw corpus on your own machine — the GPU box is
rented, shared infrastructure, and the scrubbed pairs are all the training needs.

## Memory notes at 1–3B, bf16, LoRA

| Setting | Effect |
|---|---|
| Qwen2.5-3B bf16 weights | ~6 GB |
| LoRA r=16 all-linear | ~30–60 MB of trainable params |
| `gradient_checkpointing: true` | trades ~20–30% speed for a large activation saving |
| `max_seq_len: 1024`, batch 2 × accum 8 | comfortable on 24 GB with headroom |

If you OOM: drop `per_device_batch_size` to 1 and double `grad_accum` (the effective batch is what
matters), then lower `max_seq_len` to 768. Only reach for `load_in_4bit: true` if you're on a
16 GB card or have moved past ~7B — at this size 4-bit costs quality for nothing.

## Speed

`pip install unsloth` and swapping the model load for Unsloth's `FastLanguageModel` is roughly 2×
faster with lower memory, drop-in for Qwen/Llama 1–3B. Worth doing before the Phase-2 ablation
grid, which is 13 training runs. Not worth doing before Phase 1, when you're still finding bugs.

## Sanity check before a real run

```bash
# 3 steps on a tiny slice -- catches a bad chat template or a padding-side bug in 60 seconds
# instead of 40 minutes.
python - <<'PY'
from wlm.config import Config
from wlm.train.stage_a_sft import train_stage_a
cfg = Config.load("configs/stage_a.yaml")
cfg.sft.epochs = 0.02
cfg.sft.eval_steps = cfg.sft.save_steps = 2
train_stage_a(cfg, outdir="runs/_smoke")
PY
```

Then generate one sample and read it. If the output has a stray chat-template token, is truncated,
or is in the wrong language, fix that before spending real compute.

## Serving

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-3B-Instruct --enable-lora \
  --lora-modules style=runs/stage_a --max-lora-rank 16
```

Adapters hot-swap per user, which is the whole point of keeping the style adapter modular and
standalone. Deleting the adapter directory deletes the personalization.
