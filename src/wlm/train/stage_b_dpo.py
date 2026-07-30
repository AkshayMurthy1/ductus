"""Stage B — on-policy DPO sharpening on top of the Stage-A adapter.

What this fixes that Stage A cannot: SFT maximizes likelihood of good answers but never sees --
and so never penalizes -- the model's own generic-voice failure mode. DPO does, because
`rejected` is sampled from the Stage-A policy itself.

Start with plain sigmoid DPO. Only reach for AV-reward-filtered pairs or a different loss type
once plain DPO has produced a number you trust; adding both at once makes the result
uninterpretable.

Conservative by construction: one epoch, LR ~5e-6 (an order of magnitude below Stage A), and the
Stage-A adapter as the frozen reference policy. DPO on a small preference set diverges fast.

Run:
    wlm train dpo --config configs/stage_b.yaml --out runs/stage_b --adapter runs/stage_a
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wlm.config import Config
from wlm.paths import PROCESSED, read_jsonl
from wlm.train.common import (
    accepted_kwargs,
    first_supported,
    load_base_model,
    load_tokenizer,
    save_run_meta,
    trainable_summary,
)


def _to_conversational(row: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Emit TRL *conversational* preference rows.

    TRL only applies the chat template when it detects conversational format, i.e. when the
    values are message lists. Plain strings are treated as pre-formatted text, so a
    `{"prompt": str, "chosen": str, "rejected": str}` dataset trains DPO on raw text with no
    `<|im_start|>` framing -- a format the model never sees at inference, making the reward and
    KL signal incomparable to Stage A's. Same system prompt as generation, for the same reason.
    """
    prompt = []
    if cfg.gen.system_prompt:
        prompt.append({"role": "system", "content": cfg.gen.system_prompt})
    prompt.append({"role": "user", "content": row["prompt"]})
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def train_stage_b(
    cfg: Config,
    *,
    outdir: str | Path,
    stage_a_adapter: str | Path,
    pairs_path: str | Path | None = None,
    val_pairs_path: str | Path | None = None,
) -> dict[str, Any]:
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs_path = Path(pairs_path or PROCESSED / "dpo.jsonl")
    rows = read_jsonl(pairs_path)
    if not rows:
        raise ValueError(f"{pairs_path} is empty -- run `wlm dpo-pairs` first")

    train_ds = Dataset.from_list([_to_conversational(r, cfg) for r in rows]).shuffle(
        seed=cfg.dpo.seed
    )
    val_ds = None
    if val_pairs_path and Path(val_pairs_path).exists():
        vrows = read_jsonl(val_pairs_path)
        if vrows:
            val_ds = Dataset.from_list([_to_conversational(r, cfg) for r in vrows])

    tok = load_tokenizer(cfg)
    base = load_base_model(cfg, for_training=True)
    # Continue training the Stage-A adapter in place. `is_trainable=True` is the difference
    # between fine-tuning it and silently freezing it -- a very quiet way to waste a GPU hour.
    model = PeftModel.from_pretrained(base, str(stage_a_adapter), is_trainable=True)

    # Reference policy. With ref_model=None and a PeftModel, TRL disables the adapter to get
    # reference logprobs -- i.e. the reference is the BASE model, not Stage A. That is not what
    # the plan wants: the point of the conservative LR is to stay near Stage A. Loading the
    # Stage-A weights as a second, frozen adapter and naming it via `ref_adapter_name` anchors
    # the KL term where it belongs, at the cost of one extra set of adapter weights (tens of MB,
    # not a second base model).
    ref_adapter_kwargs: dict[str, Any] = {}
    if cfg.dpo.reference_policy == "stage_a":
        try:
            model.load_adapter(str(stage_a_adapter), adapter_name="reference", is_trainable=False)
            ref_adapter_kwargs["ref_adapter_name"] = "reference"
            model.set_adapter("default")
        except Exception as e:
            print(
                f"[stage-b] could not load a frozen reference adapter ({type(e).__name__}: {e}). "
                "Falling back to the base model as the reference policy — Stage B is then free "
                "to drift away from Stage A, so watch the fluency probe and reward margins."
            )
    else:
        print("[stage-b] reference_policy='base': KL anchors to the base model, not Stage A.")
    print(f"[stage-b] {trainable_summary(model)}")

    dpo_kwargs: dict[str, Any] = dict(
        output_dir=str(outdir),
        beta=cfg.dpo.beta,
        loss_type=cfg.dpo.loss_type,
        num_train_epochs=cfg.dpo.epochs,
        learning_rate=cfg.dpo.lr,
        lr_scheduler_type=cfg.dpo.lr_scheduler,
        warmup_ratio=cfg.dpo.warmup_ratio,
        per_device_train_batch_size=cfg.dpo.per_device_batch_size,
        gradient_accumulation_steps=cfg.dpo.grad_accum,
        gradient_checkpointing=cfg.dpo.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=cfg.model.dtype == "bfloat16",
        max_prompt_length=cfg.dpo.max_prompt_len,
        max_length=cfg.dpo.max_len,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=cfg.dpo.eval_steps,
        save_strategy="steps",
        save_steps=cfg.dpo.save_steps,
        save_total_limit=2,
        logging_steps=cfg.dpo.logging_steps,
        seed=cfg.dpo.seed,
        report_to=[],
        **ref_adapter_kwargs,
    )
    args = DPOConfig(**accepted_kwargs(DPOConfig, dpo_kwargs, label="DPOConfig"))

    # Stage B is the stage most able to trade capability for voice, so it gets the same
    # catastrophic-forgetting tripwire Stage A has. It only fires on eval, so it needs val pairs.
    callbacks = []
    if val_ds is not None:
        from wlm.train.stage_a_sft import FluencyProbeCallback

        callbacks.append(
            FluencyProbeCallback(
                tok, cfg.eval.fluency_max_ppl_regression, outdir / "fluency_log.jsonl"
            )
        )
    else:
        print(
            "[stage-b] no val pairs, so no fluency probe will run and runs/stage_b will have no "
            "fluency_log.jsonl for `wlm eval run` to veto on. Pass --val-pairs (see "
            "`wlm dpo-pairs --val-frac`)."
        )

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=callbacks,
        **first_supported(DPOTrainer, {"processing_class": tok, "tokenizer": tok}),
    )
    trainer = DPOTrainer(**accepted_kwargs(DPOTrainer, trainer_kwargs, label="DPOTrainer"))
    result = trainer.train()
    # Save the trained adapter only. `save_model` would also write the frozen `reference/`
    # subdirectory -- a duplicate of Stage A -- into the final dir and every checkpoint.
    try:
        trainer.model.save_pretrained(str(outdir), selected_adapters=["default"])
    except TypeError:
        trainer.save_model(str(outdir))
    tok.save_pretrained(str(outdir))

    logs = trainer.state.log_history
    accs = [h["rewards/accuracies"] for h in logs if "rewards/accuracies" in h]
    margins = [h["rewards/margins"] for h in logs if "rewards/margins" in h]
    metrics = {
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "n_pairs": len(rows),
        "final_reward_accuracy": accs[-1] if accs else None,
        "final_reward_margin": margins[-1] if margins else None,
        "log_history": logs,
    }
    save_run_meta(outdir, cfg, {"stage": "B", "stage_a_adapter": str(stage_a_adapter),
                                "metrics": metrics})

    if accs and accs[-1] > 0.98:
        print(
            "[stage-b] reward accuracy is ~1.0, which usually means chosen/rejected differ on "
            "something trivial (length, formatting, a stray prefix) rather than on voice. "
            "Inspect the pairs before believing the gain."
        )
    return metrics
