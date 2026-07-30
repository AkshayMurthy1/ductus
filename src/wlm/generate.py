"""Generation: the Phase-0 few-shot prompting baseline, and adapter-based generation.

Both go through the same function so the comparison is apples to apples. If the baseline used a
different decoding config than the tuned run, the whole Phase-1 result would be an artifact of
temperature.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from wlm.config import Config
from wlm.paths import read_jsonl, write_jsonl


def load_for_generation(cfg: Config, adapter: str | Path | None):
    """Load once. `generate()` accepts the result so callers that loop don't reload a 3B model."""
    from wlm.train.common import load_base_model, load_tokenizer

    tok = load_tokenizer(cfg)
    tok.padding_side = "left"  # left padding for batched generation
    # Left truncation too: the few-shot baseline's system prompt can exceed max_seq_len, and
    # right truncation would cut off the user question and the assistant generation marker --
    # the model would continue the last exemplar instead of answering, quietly sinking the
    # baseline that Stage A is supposed to beat.
    tok.truncation_side = "left"
    model = load_base_model(cfg, for_training=False)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    model.config.use_cache = True
    return model, tok


def _pad_id(tok) -> int:
    # `pad_token_id or eos_token_id` is wrong when the pad id is legitimately 0.
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def build_fewshot_system(
    exemplars: list[str], base_system: str, *, n: int = 5, seed: int = 17
) -> str:
    """Phase-0 baseline: in-context style prompting. The thing tuning has to beat.

    Exemplars come from TRAIN only. Using blind-set exemplars would give the baseline a look at
    the test set and make the comparison meaningless.

    Deduplicated first: `build_pairs` emits several questions per chunk, so the raw response list
    contains each passage `questions_per_chunk` times, and a naive sample can show the model the
    same passage five times and call it five-shot.
    """
    rng = random.Random(seed)
    unique = list(dict.fromkeys(e.strip() for e in exemplars if e.strip()))
    if len(unique) < n:
        print(f"[baseline] only {len(unique)} distinct exemplars available for {n}-shot prompting")
    picks = rng.sample(unique, k=min(n, len(unique)))
    blocks = "\n\n".join(f"<sample>\n{p}\n</sample>" for p in picks)
    system = (
        f"{base_system}\n\nHere are samples of the author's writing. Match their grammatical "
        f"structure, sentence rhythm, paragraph shape and word choice.\n\n{blocks}"
    )
    words = len(system.split())
    if words > 1200:
        print(
            f"[baseline] the {n}-shot system prompt is ~{words} words. Confirm it fits inside "
            "model.max_seq_len minus gen.max_new_tokens, or the baseline is being truncated and "
            "the Phase-1 comparison is unfair in Stage A's favor."
        )
    return system


def generate(
    cfg: Config,
    prompts: list[dict[str, Any]],
    *,
    adapter: str | Path | None = None,
    system_prompt: str | None = None,
    n_per_prompt: int = 1,
    batch_size: int = 4,
    loaded: tuple[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    import torch

    model, tok = loaded if loaded is not None else load_for_generation(cfg, adapter)
    out: list[dict[str, Any]] = []
    system = system_prompt if system_prompt is not None else cfg.gen.system_prompt
    prompt_budget = max(256, cfg.gen.max_prompt_tokens - cfg.gen.max_new_tokens)

    for i in tqdm(range(0, len(prompts), batch_size), desc="generate"):
        batch = prompts[i : i + batch_size]
        texts = []
        for p in batch:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": p["prompt"]}
            ]
            texts.append(
                tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
        # Fail loudly rather than truncate: a silently shortened few-shot prompt is a fake
        # baseline, and a fake baseline makes every later phase's delta meaningless.
        over = [
            len(ids)
            for ids in tok(texts, add_special_tokens=False)["input_ids"]
            if len(ids) > prompt_budget
        ]
        if over:
            raise ValueError(
                f"{len(over)} prompt(s) exceed the {prompt_budget}-token budget (longest "
                f"{max(over)}). Raise gen.max_prompt_tokens, or lower gen.baseline_n_shots / "
                "data.chunk_max_words. Truncating here would quietly handicap whichever run has "
                "the longer prompt."
            )
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=prompt_budget).to(model.device)
        with torch.no_grad():
            ids = model.generate(
                **enc,
                do_sample=cfg.gen.temperature > 0,
                temperature=cfg.gen.temperature,
                top_p=cfg.gen.top_p,
                max_new_tokens=cfg.gen.max_new_tokens,
                num_return_sequences=n_per_prompt,
                pad_token_id=_pad_id(tok),
            )
        plen = enc["input_ids"].shape[1]
        for j, p in enumerate(batch):
            for k in range(n_per_prompt):
                seq = ids[j * n_per_prompt + k][plen:]
                out.append(
                    {
                        **{kk: p[kk] for kk in ("pair_id", "register", "doc_id") if kk in p},
                        "prompt": p["prompt"],
                        "completion": tok.decode(seq, skip_special_tokens=True).strip(),
                        "adapter": str(adapter) if adapter else "base",
                        "sample_idx": k,
                    }
                )
    return out


def run_generation(
    cfg: Config,
    *,
    split_path: str | Path,
    out_path: str | Path,
    adapter: str | Path | None = None,
    fewshot_from: str | Path | None = None,
    n_shots: int | None = None,
) -> int:
    """`fewshot_from` set + `adapter` unset = the Phase-0 baseline. Both unset = raw base model."""
    prompts = read_jsonl(split_path)
    system = cfg.gen.system_prompt
    if fewshot_from:
        exemplars = [r["response"] for r in read_jsonl(fewshot_from)]
        system = build_fewshot_system(
            exemplars, cfg.gen.system_prompt, n=n_shots or cfg.gen.baseline_n_shots
        )
    gens = generate(
        cfg, prompts, adapter=adapter, system_prompt=system, n_per_prompt=cfg.gen.n_per_prompt
    )
    return write_jsonl(out_path, gens)
