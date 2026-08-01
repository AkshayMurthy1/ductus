"""Stage A — LoRA SFT on instruction-backtranslated pairs.

This is the stage that reliably establishes the behavior, and it trains the model in its real
usage mode (an instruction in, a response out). Everything about the config is shaped by one
fact: the corpus is tiny. So -- 1-3 epochs, early stopping on validation, LoRA dropout, DoRA,
NEFTune, and a fluency probe at every checkpoint.

Run:
    wlm train sft --config configs/stage_a.yaml --out runs/stage_a
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wlm.config import Config
from wlm.paths import PROCESSED, read_jsonl
from wlm.train.common import (
    accepted_kwargs,
    build_lora_config,
    first_supported,
    load_base_model,
    load_tokenizer,
    loraplus_optimizer,
    save_run_meta,
    trainable_summary,
)


def _pc_rows(pairs: list[dict[str, Any]], system_prompt: str | None) -> list[dict[str, Any]]:
    from wlm.dataset import to_prompt_completion

    return [to_prompt_completion(p, system_prompt) for p in pairs]


def _raw_rows(
    pairs: list[dict[str, Any]], frac: float, seed: int, system_prompt: str | None = None
) -> list[dict[str, Any]]:
    """Plan §4.5's low-weight raw next-token mix, as self-continuation rows.

    Sampled with an rng rather than taken as a prefix: `build_pairs` emits several pairs per
    chunk, so `pairs[:n]` would hand the same handful of passages to every seed and double them
    up — extra memorization pressure on the most memorization-prone part of the recipe.
    """
    if frac <= 0:
        return []
    import random

    from wlm.dataset import to_self_continuation

    # Deduplicate by source chunk first: a passage appearing twice here is trained on twice.
    seen: set[str] = set()
    unique = []
    for p in pairs:
        key = p.get("chunk_id") or p["response"][:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    rng = random.Random(seed)
    n = max(1, int(len(pairs) * frac))
    rows = []
    for p in rng.sample(unique, k=min(n, len(unique))):
        row = to_self_continuation(p, system_prompt=system_prompt)
        if row:
            rows.append(row)
    return rows


class FluencyProbeCallback:
    """Fluency probe at every eval -- the catastrophic-forgetting tripwire.

    Defined as a plain class and adapted to TrainerCallback at construction time so this module
    imports cleanly on a machine with no transformers installed.
    """

    def __new__(cls, tokenizer, max_regression: float, log_path: Path):
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def on_evaluate(self, args, state, control, model=None, **kw):  # noqa: ARG002
                if model is None:
                    return
                from wlm.eval.fluency import fluency_report

                rep = fluency_report(model, tokenizer, max_regression=max_regression)
                rep["step"] = state.global_step
                print(f"[fluency] step {state.global_step}: {rep}")
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rep) + "\n")
                if rep["verdict"].startswith("FAIL"):
                    print(
                        "[fluency] regression exceeds budget. The adapter is buying style with "
                        "capability -- lower the rank or the LR, or stop earlier."
                    )

        return _CB()


def train_stage_a(
    cfg: Config,
    *,
    outdir: str | Path,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
) -> dict[str, Any]:
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = Path(train_path or PROCESSED / "train.jsonl")
    val_path = Path(val_path or PROCESSED / "val.jsonl")

    train_pairs = read_jsonl(train_path)
    val_pairs = read_jsonl(val_path)
    if len(train_pairs) < 20:
        print(
            f"[stage-a] WARNING only {len(train_pairs)} training pairs. Raise "
            "questions_per_chunk, or ingest more documents -- below ~200 pairs the val curve is "
            "too noisy to early-stop on and you are mostly measuring seed variance."
        )

    if not val_pairs:
        raise ValueError(
            f"{val_path} is empty. Early stopping, best-checkpoint restore and the fluency probe "
            "all key off eval_loss, so training without a validation set would silently run the "
            "full epoch budget on a tiny corpus -- exactly the overfitting this recipe guards "
            "against. Fix the corpus, not this check: `wlm split` only allocates a val document "
            "once a register has >=4 documents, so add documents or set data.split_by=pair for a "
            "throwaway debugging run."
        )

    # The system prompt is included in training rows because `generate()` always includes it.
    # Training on a prompt format the model never sees at inference is a silent format mismatch,
    # and completion-only loss makes including it free.
    system = cfg.gen.system_prompt
    rows = _pc_rows(train_pairs, system) + _raw_rows(
        train_pairs, cfg.sft.raw_next_token_mix, cfg.sft.seed, system
    )
    train_ds = Dataset.from_list(rows).shuffle(seed=cfg.sft.seed)
    val_ds = Dataset.from_list(_pc_rows(val_pairs, system))

    tok = load_tokenizer(cfg)
    model = load_base_model(cfg, for_training=True)
    if cfg.model.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.sft.gradient_checkpointing
        )
    model = get_peft_model(model, build_lora_config(cfg))
    summary = trainable_summary(model)
    print(f"[stage-a] {summary}")
    model.print_trainable_parameters()

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(outdir),
        num_train_epochs=cfg.sft.epochs,
        learning_rate=cfg.sft.lr,
        lr_scheduler_type=cfg.sft.lr_scheduler,
        warmup_ratio=cfg.sft.warmup_ratio,
        per_device_train_batch_size=cfg.sft.per_device_batch_size,
        per_device_eval_batch_size=cfg.sft.per_device_batch_size,
        gradient_accumulation_steps=cfg.sft.grad_accum,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=cfg.sft.weight_decay,
        max_grad_norm=cfg.sft.max_grad_norm,
        bf16=cfg.model.dtype == "bfloat16",
        fp16=cfg.model.dtype == "float16",
        packing=False,  # packing mixes authors' passages across boundaries; don't
        neftune_noise_alpha=cfg.sft.neftune_noise_alpha,
        # Loss on the completion only: we are learning how the person writes, and the
        # backtranslated question is machine-written text we explicitly do not want to imitate.
        # This works because the dataset is in prompt-completion form (see to_prompt_completion) --
        # no `{% generation %}`-annotated chat template required.
        completion_only_loss=cfg.sft.train_on_completions_only,
        eval_strategy="steps",
        eval_steps=cfg.sft.eval_steps,
        save_strategy="steps",
        save_steps=cfg.sft.save_steps,
        save_total_limit=cfg.sft.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg.sft.logging_steps,
        seed=cfg.sft.seed,
        report_to=[],
        dataset_num_proc=1,
    )
    # `max_seq_length` was renamed `max_length`; take whichever this trl has.
    sft_kwargs.update(
        first_supported(
            SFTConfig,
            {"max_length": cfg.model.max_seq_len, "max_seq_length": cfg.model.max_seq_len},
        )
    )
    args = SFTConfig(**accepted_kwargs(SFTConfig, sft_kwargs, label="SFTConfig"))
    if cfg.sft.train_on_completions_only and not hasattr(args, "completion_only_loss"):
        print(
            "[compat] this trl version has no `completion_only_loss`, so loss is computed over "
            "the prompt as well. The backtranslated questions are machine-written, so training "
            "on them dilutes the style signal — upgrade trl, or wrap the collator in "
            "DataCollatorForCompletionOnlyLM before trusting Phase-1 numbers."
        )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=cfg.sft.early_stopping_patience),
        FluencyProbeCallback(
            tok, cfg.eval.fluency_max_ppl_regression, outdir / "fluency_log.jsonl"
        ),
    ]

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
        optimizers=(loraplus_optimizer(model, cfg), None),
        # `tokenizer` was renamed `processing_class`.
        **first_supported(SFTTrainer, {"processing_class": tok, "tokenizer": tok}),
    )
    trainer = SFTTrainer(**accepted_kwargs(SFTTrainer, trainer_kwargs, label="SFTTrainer"))

    result = trainer.train()
    trainer.save_model(str(outdir))
    tok.save_pretrained(str(outdir))

    metrics = {
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "train_loss": result.metrics.get("train_loss"),
        "log_history": trainer.state.log_history,
        "trainable": summary,
        "n_train_rows": len(rows),
        "n_val_rows": len(val_ds),
    }
    save_run_meta(outdir, cfg, {"stage": "A", "metrics": metrics})

    # Overfitting read-out you should actually look at.
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    if len(evals) >= 3:
        best = min(evals, key=lambda h: h["eval_loss"])
        last = evals[-1]
        if last["eval_loss"] > best["eval_loss"] * 1.05:
            print(
                f"[stage-a] val loss rose from {best['eval_loss']:.4f} (step {best['step']}) to "
                f"{last['eval_loss']:.4f} -- best checkpoint restored. Consider fewer epochs."
            )
    return metrics
