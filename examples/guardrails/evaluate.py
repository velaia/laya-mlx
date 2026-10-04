"""Evaluate the guardrail classifier against public safety/jailbreak datasets.

Datasets expected under --data-dir (default ~/data/guardrail-datasets/):
  wildguardmix              -- primary multi-class benchmark
  aegis-2.0                 -- secondary safe/harmful_content/privacy_or_fraud check
  deepset-prompt-injections -- jailbreak-only precision/recall (not content-labeled otherwise)
  jbb-behaviors             -- harmful-vs-safe sanity check

Each dataset's own, finer-grained labels are mapped onto our 4 categories
(safe / harmful_content / privacy_or_fraud / jailbreak). Labels that don't
map cleanly (misinformation, copyright, political content, expert-advice,
government decision-making, ...) are marked unmapped and EXCLUDED from
scoring rather than force-fit into one of our buckets.

wildguardmix rows flagged `adversarial=True` are scored as expected
"jailbreak" regardless of their harm subcategory, since our taxonomy treats
"uses a jailbreak technique" and "content is harmful" as a single, mutually
exclusive choice -- this is a modeling decision, not a property of the data,
and it means harmful_content recall is only measured on non-adversarial rows.

Requires pandas + pyarrow for the parquet datasets:
    uv run --with pandas --with pyarrow examples/guardrails/evaluate.py

Usage:
    uv run --with pandas --with pyarrow examples/guardrails/evaluate.py
    uv run --with pandas --with pyarrow examples/guardrails/evaluate.py --dataset wildguardmix
    uv run --with pandas --with pyarrow examples/guardrails/evaluate.py --limit 200
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import laya_mlx as laya
import ollama_client

ROOT = Path(__file__).parent
QUESTIONS = json.loads((ROOT / "questions.json").read_text())
CATEGORIES = ["safe", "harmful_content", "privacy_or_fraud", "jailbreak"]


def classify(agent, text):
    return agent.predict(text, QUESTIONS)["answers"]["category"]["choice"]


# ---- dataset loaders: each returns a list of {"text": ..., "expected": ...} ----

WILDGUARD_HARMFUL = {
    "violence_and_physical_harm",
    "toxic_language_hate_speech",
    "sexual_content",
    "mental_health_over-reliance_crisis",
    "social_stereotypes_and_unfair_discrimination",
    "defamation_encouraging_unethical_or_unsafe_actions",
    "cyberattack",
}
WILDGUARD_PRIVACY_FRAUD = {
    "private_information_individual",
    "sensitive_information_organization_government",
    "fraud_assisting_illegal_activities",
}


def load_wildguardmix(data_dir, limit=None):
    import pandas as pd

    df = pd.read_parquet(data_dir / "wildguardmix" / "test" / "wildguard_test.parquet")
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    rows = []
    for r in df.itertuples():
        if r.adversarial:
            expected = "jailbreak"
        elif r.subcategory == "benign":
            expected = "safe"
        elif r.subcategory in WILDGUARD_HARMFUL:
            expected = "harmful_content"
        elif r.subcategory in WILDGUARD_PRIVACY_FRAUD:
            expected = "privacy_or_fraud"
        else:
            expected = None
        rows.append({"text": r.prompt, "expected": expected})
    return rows


AEGIS_PRIVACY_FRAUD = {"PII/Privacy", "Fraud/Deception"}
AEGIS_HARMFUL = {
    "Violence",
    "Threat",
    "Guns and Illegal Weapons",
    "Controlled/Regulated Substances",
    "Suicide and Self Harm",
    "Hate/Identity Hate",
    "Harassment",
    "Sexual",
    "Sexual (minor)",
    "Profanity",
    "Immoral/Unethical",
    "Illegal Activity",
    "Malware",
    "Criminal Planning/Confessions",
}


def load_aegis(data_dir, limit=None):
    rows_raw = json.loads((data_dir / "aegis-2.0" / "test.json").read_text())
    if limit:
        import random
        rows_raw = random.Random(42).sample(rows_raw, min(limit, len(rows_raw)))
    rows = []
    for r in rows_raw:
        if r["prompt_label"] == "safe":
            expected = "safe"
        else:
            cats = {c.strip() for c in (r.get("violated_categories") or "").split(",")}
            if cats & AEGIS_PRIVACY_FRAUD:
                expected = "privacy_or_fraud"
            elif cats & AEGIS_HARMFUL:
                expected = "harmful_content"
            else:
                expected = None
        rows.append({"text": r["prompt"], "expected": expected})
    return rows


def load_deepset(data_dir, limit=None):
    import pandas as pd

    df = pd.read_parquet(
        data_dir / "deepset-prompt-injections" / "data" / "test-00000-of-00001-701d16158af87368.parquet"
    )
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)
    return [{"text": r.text, "expected_is_jailbreak": bool(r.label)} for r in df.itertuples()]


JBB_PRIVACY_FRAUD = {"Privacy", "Fraud/Deception", "Economic harm"}
JBB_HARMFUL = {"Harassment/Discrimination", "Physical harm", "Sexual/Adult content", "Malware/Hacking"}


def load_jbb(data_dir, limit=None):
    import pandas as pd

    base = data_dir / "jbb-behaviors" / "data"
    harmful = pd.read_csv(base / "harmful-behaviors.csv")
    benign = pd.read_csv(base / "benign-behaviors.csv")
    rows = []
    for r in harmful.itertuples():
        if r.Category in JBB_PRIVACY_FRAUD:
            expected = "privacy_or_fraud"
        elif r.Category in JBB_HARMFUL:
            expected = "harmful_content"
        else:
            expected = None
        rows.append({"text": r.Goal, "expected": expected})
    for r in benign.itertuples():
        rows.append({"text": r.Goal, "expected": "safe"})
    if limit:
        import random
        rows = random.Random(42).sample(rows, min(limit, len(rows)))
    return rows


# ---- scoring ----


def _prf1(tp, support, predicted_count):
    precision = tp / predicted_count if predicted_count else math.nan
    recall = tp / support if support else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_multiclass(name, rows, agent):
    confusion = defaultdict(Counter)  # confusion[expected][predicted]
    unmapped = 0
    for row in rows:
        expected = row["expected"]
        predicted = classify(agent, row["text"])
        if expected is None:
            unmapped += 1
            continue
        confusion[expected][predicted] += 1

    total = sum(sum(c.values()) for c in confusion.values())
    correct = sum(confusion[c][c] for c in CATEGORIES)
    print(f"\n== {name} ==")
    print(f"scored: {total}  (unmapped/excluded: {unmapped})")
    if total:
        print(f"accuracy: {correct / total:.3f}")
    print(f"{'class':<18}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>7}")
    for c in CATEGORIES:
        support = sum(confusion[c].values())
        tp = confusion[c][c]
        predicted_c = sum(confusion[e][c] for e in CATEGORIES)
        precision, recall, f1 = _prf1(tp, support, predicted_c)
        print(f"{c:<18}{support:>8}{precision:>11.3f}{recall:>9.3f}{f1:>7.3f}")
    print("confusion (rows=expected, cols=predicted):")
    print(" " * 18 + "".join(f"{c[:10]:>12}" for c in CATEGORIES))
    for e in CATEGORIES:
        print(f"{e:<18}" + "".join(f"{confusion[e][p]:>12}" for p in CATEGORIES))


def score_jailbreak_binary(name, rows, agent):
    tp = fp = tn = fn = 0
    for row in rows:
        predicted_jb = classify(agent, row["text"]) == "jailbreak"
        expected_jb = row["expected_is_jailbreak"]
        if predicted_jb and expected_jb:
            tp += 1
        elif predicted_jb and not expected_jb:
            fp += 1
        elif not predicted_jb and expected_jb:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    precision, recall, f1 = _prf1(tp, tp + fn, tp + fp)
    print(f"\n== {name} (jailbreak detection only) ==")
    print(f"scored: {total}")
    print(f"accuracy: {(tp + tn) / total:.3f}")
    print(f"precision: {precision:.3f}  recall: {recall:.3f}  f1: {f1:.3f}")
    print(f"tp={tp} fp={fp} tn={tn} fn={fn}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(Path.home() / "data" / "guardrail-datasets"))
    parser.add_argument("--model", default="aac6fef/laya-mlx")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows per dataset (default: full test split).")
    parser.add_argument(
        "--dataset",
        choices=["wildguardmix", "aegis-2.0", "deepset-prompt-injections", "jbb-behaviors", "all"],
        default="all",
    )
    parser.add_argument(
        "--backend",
        choices=["laya", "ollama", "openrouter"],
        default="laya",
        help="laya: laya-mlx checkpoint. ollama: a decision model served via Ollama's /v1/systemone (clef, clef-flash, ...). "
        "openrouter: a System One model on OpenRouter (~typesafe/jev-latest, ...), needs OPENROUTER_API_KEY.",
    )
    parser.add_argument("--base-url", default=None, help="Server URL override (default: localhost Ollama / OpenRouter).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    if args.backend != "laya":
        agent = ollama_client.load_backend(args.backend, args.model, args.base_url)
    else:
        agent = laya.load(args.model)

    if args.dataset in ("wildguardmix", "all"):
        score_multiclass("wildguardmix", load_wildguardmix(data_dir, args.limit), agent)
    if args.dataset in ("aegis-2.0", "all"):
        score_multiclass("aegis-2.0", load_aegis(data_dir, args.limit), agent)
    if args.dataset in ("deepset-prompt-injections", "all"):
        score_jailbreak_binary("deepset-prompt-injections", load_deepset(data_dir, args.limit), agent)
    if args.dataset in ("jbb-behaviors", "all"):
        score_multiclass("jbb-behaviors", load_jbb(data_dir, args.limit), agent)


if __name__ == "__main__":
    sys.exit(main())
