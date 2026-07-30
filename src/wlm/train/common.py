"""Shared model/adapter loading. GPU-only imports live here, lazily."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from wlm.config import Config


def accepted_kwargs(target, kwargs: dict[str, Any], *, label: str = "") -> dict[str, Any]:
    """Drop kwargs the installed version of `target` doesn't accept, loudly.

    `trl`/`transformers` rename arguments between minor versions (`max_seq_length` ->
    `max_length`, `tokenizer` -> `processing_class`, `torch_dtype` -> `dtype`,
    `assistant_only_loss` appearing only recently). Silently passing an unknown kwarg is a
    TypeError that reads like a code bug; silently *dropping* one changes the experiment without
    telling you. So: drop, but print what was dropped, so a missing knob shows up in the log
    next to the run it affected.
    """
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    ok = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(ok))
    if dropped:
        name = label or getattr(target, "__name__", str(target))
        print(
            f"[compat] {name} in your installed version does not accept {dropped} — dropped. "
            "If one of those was a knob you meant to set, upgrade the library or set the "
            "equivalent argument for your version."
        )
    return ok


def first_supported(target, candidates: dict[str, Any]) -> dict[str, Any]:
    """Pick whichever alias the installed version supports, e.g. max_length vs max_seq_length.

    Order the candidates newest-name-first. If the target takes `**kwargs` we cannot introspect
    which alias it honors, so the first candidate wins -- returning {} there would silently drop
    the argument entirely, which is how a model ends up loading in fp32 and OOMing a 24 GB card.
    """
    if not candidates:
        return {}
    first = next(iter(candidates.items()))
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return dict([first])
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict([first])
    for name, value in candidates.items():
        if name in params:
            return {name: value}
    return {}


def torch_dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg.model.name, trust_remote_code=cfg.model.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding for training, left for generation -- get this backwards and generations
    # come out truncated in ways that look like a style problem.
    tok.padding_side = "right"
    return tok


def load_base_model(cfg: Config, *, for_training: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "attn_implementation": cfg.model.attn_implementation,
        "trust_remote_code": cfg.model.trust_remote_code,
        # `torch_dtype` was renamed to `dtype`; take whichever this transformers has.
        **first_supported(
            AutoModelForCausalLM.from_pretrained,
            {"dtype": torch_dtype(cfg.model.dtype), "torch_dtype": torch_dtype(cfg.model.dtype)},
        ),
    }
    if cfg.model.load_in_4bit:
        # Only needed if you scale past ~7B or drop to a 16 GB card. At 1-3B, bf16 LoRA fits in
        # 24 GB with room to spare and 4-bit quantization costs quality for nothing.
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            # Follow cfg.model.dtype so this doesn't silently contradict the fp16/bf16 flag
            # passed to the Trainer.
            bnb_4bit_compute_dtype=torch_dtype(cfg.model.dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # A 4-bit model must be placed by `device_map` at load time; calling `.to()` on one
        # raises. So always give the quantized path an explicit map and never move it after.
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "cpu"
    elif not for_training:
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **kwargs)

    if for_training and not cfg.model.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")

    dtype_seen = next(model.parameters()).dtype
    if not cfg.model.load_in_4bit and str(dtype_seen) != f"torch.{cfg.model.dtype}":
        print(
            f"[compat] requested {cfg.model.dtype} but the model loaded as {dtype_seen}. On a "
            "24 GB card a 3B model in float32 will OOM — check that transformers accepted the "
            "dtype argument."
        )
    model.config.use_cache = not (for_training and cfg.sft.gradient_checkpointing)
    return model


def build_lora_config(cfg: Config):
    """LoRA on attention AND MLP.

    Attention (q/k/v/o) shapes structure, agreement and flow. The MLP/SwiGLU projections
    (gate/up/down) carry lexical choice, idiom and phrasing. Attention alone captures cadence
    but underfits word choice, which is where a person's characteristic phrasings live -- so
    both, via `all-linear`.

    The MLP is also where topic memorization shows up, because it holds the most parameters. The
    fix is a lower MLP rank (`rank_pattern`), not dropping the MLP.
    """
    from peft import LoraConfig

    kwargs: dict[str, Any] = {
        "r": cfg.lora.r,
        "lora_alpha": cfg.lora.lora_alpha,
        "lora_dropout": cfg.lora.lora_dropout,
        "bias": cfg.lora.bias,
        "task_type": "CAUSAL_LM",
        "target_modules": cfg.lora.target_modules,
        "use_dora": cfg.lora.use_dora,
        "use_rslora": cfg.lora.use_rslora,
    }
    if isinstance(cfg.lora.target_modules, str) and cfg.lora.target_modules != "all-linear":
        raise ValueError(
            "target_modules must be the string 'all-linear' or an explicit list of module names, "
            f"got {cfg.lora.target_modules!r}"
        )
    if cfg.lora.rank_pattern:
        kwargs["rank_pattern"] = cfg.lora.rank_pattern
    if cfg.lora.alpha_pattern:
        kwargs["alpha_pattern"] = cfg.lora.alpha_pattern
    if cfg.lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = cfg.lora.layers_to_transform
    if cfg.lora.modules_to_save:
        kwargs["modules_to_save"] = cfg.lora.modules_to_save
    return LoraConfig(**kwargs)


def loraplus_optimizer(model, cfg: Config):
    """LoRA+: a higher LR on the B matrices than on A. Near-free quality gain.

    Returns None when disabled so the caller falls back to the Trainer's default optimizer.
    """
    if not cfg.sft.loraplus_lr_ratio:
        return None
    import torch

    a_params, b_params, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name:
            a_params.append(p)
        elif "lora_B" in name:
            b_params.append(p)
        else:
            other.append(p)
    groups = [
        {"params": a_params, "lr": cfg.sft.lr},
        {"params": b_params, "lr": cfg.sft.lr * cfg.sft.loraplus_lr_ratio},
        {"params": other, "lr": cfg.sft.lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=cfg.sft.weight_decay)


def trainable_summary(model, *, exclude: str = "reference") -> dict[str, Any]:
    """Parameter counts. `exclude` drops a named adapter (Stage B's frozen reference copy) so
    trainable_pct stays comparable across stages."""
    def keep(name: str) -> bool:
        return f".{exclude}." not in name and not name.endswith(f".{exclude}")

    total = sum(p.numel() for n, p in model.named_parameters() if keep(n))
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and keep(n))
    by_kind = {"attention": 0, "mlp": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad or not keep(name):
            continue
        if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            by_kind["attention"] += p.numel()
        elif any(k in name for k in ("gate_proj", "up_proj", "down_proj")):
            by_kind["mlp"] += p.numel()
        else:
            by_kind["other"] += p.numel()
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / max(1, total), 4),
        "trainable_by_kind": by_kind,
    }


def save_run_meta(outdir: str | Path, cfg: Config, extra: dict[str, Any]) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_meta.json").write_text(
        json.dumps({"config": cfg.to_dict(), **extra}, indent=2, default=str), encoding="utf-8"
    )
