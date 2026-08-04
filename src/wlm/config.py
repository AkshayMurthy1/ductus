"""Typed config loaded from configs/*.yaml. Keep every knob here so ablations are one file diff."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelCfg:
    # Plan §2: 1-3B open-weight instruct model. Open weights are required -- LoRA on chosen
    # matrices, style-vector steering and DPO all need direct weight access.
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    # bf16 at 1-3B on 24 GB fits comfortably (~6 GB of weights); QLoRA is only for scaling up.
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str = "sdpa"
    max_seq_len: int = 1024
    trust_remote_code: bool = False


@dataclass
class LoraCfg:
    # "all-linear" hits q/k/v/o AND gate/up/down. Attention shapes cadence and agreement;
    # the MLP carries lexical choice and idiom. You need both (plan §3).
    target_modules: str | list[str] = "all-linear"
    r: int = 16
    lora_alpha: int = 32  # alpha ~= 2*r
    lora_dropout: float = 0.05
    bias: str = "none"
    # DoRA: decomposes update into magnitude + direction. Wins at low rank / small data.
    use_dora: bool = True
    use_rslora: bool = False
    # Lower the MLP rank rather than dropping the MLP if ablations show topic leak.
    # e.g. {"gate_proj": 8, "up_proj": 8, "down_proj": 8}
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, int] = field(default_factory=dict)
    # Optional layer trimming ablation; None = all layers.
    layers_to_transform: list[int] | None = None
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class SftCfg:
    epochs: float = 2.0
    lr: float = 1.5e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    per_device_batch_size: int = 2
    grad_accum: int = 8  # effective batch 16
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    # NEFTune helps most in exactly this small-data regime.
    neftune_noise_alpha: float | None = 5.0
    # LoRA+: higher LR on the B matrices. 0 disables.
    loraplus_lr_ratio: float = 0.0
    # Loss on the completion only -- we are learning how the person writes, not the questions.
    train_on_completions_only: bool = True
    # Fraction of steps that are raw next-token modeling on real passages (plan §4.5).
    raw_next_token_mix: float = 0.1
    eval_steps: int = 25
    save_steps: int = 25
    # Raise for a checkpoint-trajectory run (scripts/05_checkpoint_trajectory.sh), which needs
    # every checkpoint kept so the style-vs-leakage frontier can be traced across training time.
    save_total_limit: int = 3
    logging_steps: int = 5
    early_stopping_patience: int = 3
    seed: int = 17


@dataclass
class DpoCfg:
    beta: float = 0.1
    epochs: float = 1.0
    lr: float = 5e-6
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    per_device_batch_size: int = 1
    grad_accum: int = 8
    gradient_checkpointing: bool = True
    eval_steps: int = 25
    save_steps: int = 25
    logging_steps: int = 5
    max_prompt_len: int = 512
    max_len: int = 1024
    loss_type: str = "sigmoid"  # plain DPO first; upgrade later
    # Reference policy. "stage_a" loads a second frozen copy of the Stage-A adapter so the KL
    # term anchors to Stage A (what the plan intends). "base" uses the adapter-disabled base
    # model, which lets Stage B walk away from Stage A -- cheaper in memory, weaker guarantee.
    reference_policy: str = "stage_a"
    # Drop pairs where the AV score already prefers the model's own generation.
    av_reward_filter: bool = False
    av_margin: float = 0.05
    # Match chosen/rejected length so DPO learns voice, not formatting artifacts.
    length_match_tolerance: float = 0.35
    seed: int = 17


@dataclass
class GenCfg:
    temperature: float = 0.8
    top_p: float = 0.95
    # Sampling seed. Threaded so a variance cell (same config, different seed) is a CLI flag
    # rather than a YAML edit -- see RESEARCH_BRIEF §3 "Variance".
    seed: int = 17
    max_new_tokens: int = 400
    n_per_prompt: int = 1
    # Input cap at generation time. Deliberately NOT model.max_seq_len: that is a *training*
    # length chosen for memory, while Qwen2.5 serves 32k. Reusing it would silently truncate the
    # few-shot baseline's system prompt and hand Stage A a win it didn't earn.
    max_prompt_tokens: int = 8192
    # Few-shot prompting baseline (Phase 0) -- the thing tuning has to beat.
    baseline_n_shots: int = 5
    system_prompt: str = (
        "You are writing as the author. Match the author's grammatical structure, sentence "
        "rhythm, paragraph flow, and word choice. Do not imitate the author's topics or "
        "invent facts about their life."
    )


@dataclass
class DataCfg:
    chunk_min_words: int = 60
    chunk_max_words: int = 320
    questions_per_chunk: int = 2
    # When a document carries the real prompt it was written to answer, prefer it over a
    # generated question -- a generated one is a guess, the real one is ground truth.
    use_supplied_prompts: bool = True
    # "first_chunk": only the opening chunk answers the prompt; later chunks are mid-passage and
    # get generated questions instead. "all_chunks" pairs every chunk with the prompt.
    supplied_prompt_scope: str = "first_chunk"
    val_frac: float = 0.15
    blind_frac: float = 0.15
    # A passage reused across two prompts becomes two documents; splitting by document then
    # leaks it into blind. Group documents sharing more than this fraction of 5-grams.
    group_near_duplicates: bool = True
    near_dup_threshold: float = 0.15
    # Terms the NER mistakes for names but which carry meaning, not identity. Adds to
    # scrub.DEFAULT_NEVER_SCRUB; put your field's eponyms and tool names here.
    never_scrub: list[str] = field(default_factory=list)
    # Split by document, not by chunk -- chunk-level splits leak an author's document voice
    # across the boundary and inflate every number you care about.
    split_by: str = "document"
    scrub_entities: bool = True
    seed: int = 17


@dataclass
class EvalCfg:
    style_embedder: str = "AnnaWegmann/Style-Embedding"
    # Second, independently-trained style instrument (scripts/second_instrument.py). Every
    # headline number should agree under both rulers before publication; a claim only one
    # embedder supports is a claim about that embedder.
    second_style_embedder: str = "StyleDistance/styledistance"
    av_classifier: str = "logreg"
    fluency_max_ppl_regression: float = 0.15  # fail the run if ppl worsens >15%
    informal_tag: str = "informal"


@dataclass
class Config:
    run_name: str = "stage_a"
    notes: str = ""
    model: ModelCfg = field(default_factory=ModelCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    sft: SftCfg = field(default_factory=SftCfg)
    dpo: DpoCfg = field(default_factory=DpoCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    data: DataCfg = field(default_factory=DataCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return _from_dict(cls, _load_raw(Path(path)))

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _load_raw(path: Path, _seen: frozenset[Path] = frozenset()) -> dict:
    """Resolve `_extends` chains recursively (a15 extends stage_b, which extends stage_a),
    nearest file winning each key. Cycles fail loudly rather than recursing forever."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"_extends cycle through {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("_extends", None)
    if base:
        return _deep_merge(_load_raw(path.parent / base, _seen | {path}), raw)
    return raw


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(cls, data: dict[str, Any]):
    # `from __future__ import annotations` makes field.type a string, so resolve properly.
    hints = get_type_hints(cls)
    kwargs = {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name in known:
        if name not in data:
            continue
        val = data[name]
        typ = hints.get(name)
        if is_dataclass(typ) and isinstance(val, dict):
            kwargs[name] = _from_dict(typ, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _to_dict(obj) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
