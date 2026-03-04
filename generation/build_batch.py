"""Build a GPT batch input JSONL for Arabic essay generation.

Supports 5 prompting conditions (P1-P5):
  P1 — no level, no vocab, no profile
  P2 — level only
  P3 — level + linguistic profile (no vocab)
  P4 — level + vocab
  P5 — level + linguistic profile + vocab

P4 and P5 require --vocab_csv.
P3 and P5 require --profile_csv.

Input CSV (--prompts_csv) must contain columns:
  prompt_id, CEFR_level, Arabic Text

Output JSONL (--out_jsonl) is ready for OpenAI batch API upload.

Example:
  # P2 — level only:
  python generation/build_batch.py \
    --prompts_csv <prompts_csv> \
    --out_jsonl <out_jsonl> \
    --prompt_version P2

  # P5 — full profile + vocab:
  python generation/build_batch.py \
    --prompts_csv <prompts_csv> \
    --profile_csv <profile_csv> \
    --vocab_csv <vocab_csv> \
    --out_jsonl <out_jsonl> \
    --prompt_version P5 \
    --model gpt-4o \
    --custom_id_suffix 6levels_p5
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_MSG = "أنت مساعد يكتب مقالات إنشائية عربية بمستويات قرائية مختلفة."

TREE_DEPTH_MAP: Dict[str, Tuple[int, int]] = {
    "A":  (4,  8),
    "B":  (5,  8),
    "C":  (11, 40),
    "A1": (4,  6),
    "A2": (6,  8),
    "B1": (7,  10),
    "B2": (8,  11),
    "C1": (9,  12),
    "C2": (9,  13),
}

# Profile keys required by P3 and P5
PROFILE_KEYS = [
    "total_words", "avg_words_per_sentence", "total_unique_words",
    "overall_avg_word_len", "pos_ADJ_mean", "dep_SBJ_mean", "dep_OBJ_mean",
]

_NAN_TOKEN = re.compile(r"(?<![\w])nan(?![\w])", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_words_cell(s: str) -> list:
    """Parse a string-serialised list of word tuples from a vocab CSV cell.

    Args:
        s: String representation of a Python list.

    Returns:
        Parsed list, or empty list on parse failure.
    """
    s = _NAN_TOKEN.sub("None", str(s).strip())
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


def load_profiles(profile_csv: str) -> Dict[str, dict]:
    """Load CEFR-level linguistic profiles from a CSV.

    Handles the special case of level 'A' by averaging A1 and A2 profiles.

    Args:
        profile_csv: Path to the profile CSV with a 'level' column.

    Returns:
        Dict mapping CEFR level string to a profile dict.

    Raises:
        ValueError: If required profile keys are missing.
    """
    df = pd.read_csv(profile_csv)
    if "level" not in df.columns:
        raise ValueError(
            f"Profile CSV must contain a 'level' column. "
            f"Found: {list(df.columns)}"
        )

    profiles = df.set_index("level").to_dict(orient="index")

    # Synthesize 'A' as average of A1 and A2 if not explicitly present
    if "A" not in profiles and "A1" in profiles and "A2" in profiles:
        profiles["A"] = {
            k: round((profiles["A1"].get(k, 0) + profiles["A2"].get(k, 0)) / 2, 2)
            for k in PROFILE_KEYS
        }

    return profiles


def load_vocab_map(vocab_csv: str) -> Dict[Tuple[str, str], list]:
    """Load vocab words keyed by (prompt_id, CEFR_level).

    Args:
        vocab_csv: Path to vocab CSV with columns: prompt_id, CEFR_level, words.

    Returns:
        Dict mapping (prompt_id, CEFR_level) -> list of word tuples.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(vocab_csv, dtype=str)
    required = {"prompt_id", "CEFR_level", "words"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Vocab CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )
    vocab_map = {}
    for _, row in df.iterrows():
        key             = (str(row["prompt_id"]), str(row["CEFR_level"]))
        vocab_map[key]  = parse_words_cell(row.get("words", "[]"))
    return vocab_map


# ---------------------------------------------------------------------------
# Prompt builders (P1 – P5)
# ---------------------------------------------------------------------------

def build_p1(topic_text: str, **_) -> str:
    """P1: No level, no vocab, no profile."""
    return (
        f"اكتب مقالًا إنشائيًا عربيًا متكاملًا.\n"
        f"يجب أن يتناول المقال الموضوع التالي: {topic_text}.\n"
        "يجب أن يكون النص مترابطًا وسلسًا في فقرة أو عدة فقرات متتابعة، "
        "ويعكس بنية منطقية تبدأ بتمهيد للفكرة ثم تطويرها ثم إنهائها بخلاصة طبيعية، "
        "من دون استخدام عناوين أو كلمات دالة مثل (مقدمة، محتوى، خاتمة) أو أي تقسيم صريح."
    )


def build_p2(topic_text: str, level: str, **_) -> str:
    """P2: Level label only."""
    return (
        f"اكتب مقالًا إنشائيًا عربيًا متكاملًا بمستوى قابلية قراءة {level}.\n"
        f"يجب أن يتناول المقال الموضوع التالي: {topic_text}.\n"
        "يجب أن يكون النص مترابطًا وسلسًا في فقرة أو عدة فقرات متتابعة، "
        "ويعكس بنية منطقية تبدأ بتمهيد للفكرة ثم تطويرها ثم إنهائها بخلاصة طبيعية، "
        "من دون استخدام عناوين أو كلمات دالة مثل (مقدمة، محتوى، خاتمة) أو أي تقسيم صريح."
    )


def build_p3(topic_text: str, level: str, profile: dict,
             depth_min: int, depth_max: int, **_) -> str:
    """P3: Level + detailed linguistic profile, no vocab."""
    return (
        f"اكتب مقالًا إنشائيًا متكاملًا بمستوى قابلية قراءة {level}.\n"
        f"يجب أن يكون المقال عن الموضوع التالي: {topic_text}.\n\n"
        "حاول أن يكون النص قريبًا من الخصائص العامة التالية:\n\n"
        f"1) الطول والبنية:\n"
        f"- طول النص قريب من {round(profile['total_words'])}.\n"
        f"- طول الجمل قريب من {round(profile['avg_words_per_sentence'])}.\n"
        f"- عدد الكلمات الفريدة قريب من {round(profile['total_unique_words'])}.\n"
        f"- متوسط طول الكلمة (عدد الحروف): {round(profile['overall_avg_word_len'])}\n\n"
        f"3) الوصف والتفصيل:\n"
        f"- عدد الصفات: {round(profile['pos_ADJ_mean'])}.\n"
        "- استخدم صفات لوصف الأشخاص والأشياء والأفكار.\n"
        "- كلما زاد المستوى، زادت التفاصيل.\n\n"
        f"4) بناء الجملة والمعنى:\n"
        f"- عدد الفاعل (قريب من {round(profile['dep_SBJ_mean'])}).\n"
        f"- عدد مفاعيل وأمثلة (قريب من {round(profile['dep_OBJ_mean'])}).\n"
        "- في المستويات المتقدمة(B1-C2)، اربط السبب بالنتيجة.\n\n"
        f"5) البنية النحوية (شجرة التحليل):\n"
        f"- العمق الأدنى لشجرة التحليل قريب من {depth_min}.\n"
        f"- العمق الأقصى لشجرة التحليل قريب من {depth_max}.\n"
        "- كلما ارتفع المستوى، استخدم جملًا ذات بنية أعمق وأكثر تفرعًا.\n\n"
        "6) الترابط والمنطق:\n"
        "- استخدم روابط منطقية تناسب المستوى.\n"
        "- في المستويات المتقدمة، استخدم التعليل والمقارنة.\n\n"
        "7) الأسلوب العام:\n"
        "- لا أسلوب حواري إلا في B2 أو أعلى.\n"
        "- تجنّب التعجب والمبالغة في المستويات الدنيا.\n"
        "- الأسلوب يجب أن يعكس مستوى متعلم حقيقي.\n\n"
        "قيود مهمة:\n"
        "- هذه القيم إرشادية فقط.\n"
        "- لا تذكر أي أرقام في النص.\n"
        "- لا تشرح هذه القيود في الناتج.\n\n"
        "هيكل النص:\n"
        "- مقدمة، محتوى، خاتمة بدون عناوين.\n"
        "- فقرات متصلة فقط.\n"
    )


def build_p4(topic_text: str, level: str, words: list, **_) -> str:
    """P4: Level + vocab, no profile."""
    return (
        f"اكتب مقالًا إنشائيًا متكاملًا بمستوى قابلية قراءة {level}.\n"
        f"يجب أن يكون المقال عن الموضوع التالي: {topic_text}.\n"
        f"تأكد من إدخال بعض الكلمات المفتاحية التالية بشكل طبيعي داخل النص لإنشاء قصة متماسكة جميلة: {words}.\n\n"
        "يجب أن يتكوّن المقال من مقدمة، محتوى، وخاتمة، دون أي عناوين أو تقسيمات ظاهرة.\n"
        "اكتب النص كسرد متصل في فقرات متتابعة فقط، دون أي تنسيق أو عناوين فرعية.\n\n"
        "احرص على أن يكون المقال مترابطًا، منظمًا، ومناسبًا لمستوى القراءة المطلوب.\n"
        "تأكّد من أن جميع الجمل منطقية ومترابطة من حيث المعنى.\n"
        "تجنّب أي عبارات أو أفكار غير واقعية أو غير منطقية.\n"
        "تأكّد من أن الأمثلة والحقائق تتوافق مع العادات، والثقافة، والمنطق العام.\n"
        "احرص على أن يكون النص واضحًا وسهل الفهم دون أخطاء لغوية أو تركيبية.\n"
    )


def build_p5(topic_text: str, level: str, profile: dict,
             depth_min: int, depth_max: int, words: list, **_) -> str:
    """P5: Level + detailed linguistic profile + vocab."""
    return (
        f"اكتب مقالًا إنشائيًا متكاملًا بمستوى قابلية قراءة {level}.\n"
        f"يجب أن يكون المقال عن الموضوع التالي: {topic_text}.\n"
        f"أدرج الكلمات المفتاحية التالية بشكل طبيعي داخل النص: {words}.\n\n"
        "حاول أن يكون النص قريبًا من الخصائص العامة التالية:\n\n"
        f"1) الطول والبنية:\n"
        f"- طول النص قريب من {round(profile['total_words'])}.\n"
        f"- طول الجمل قريب من {round(profile['avg_words_per_sentence'])}.\n"
        f"- عدد الكلمات الفريدة قريب من {round(profile['total_unique_words'])}.\n"
        f"- متوسط طول الكلمة (عدد الحروف): {round(profile['overall_avg_word_len'])}\n\n"
        f"3) الوصف والتفصيل:\n"
        f"- عدد الصفات: {round(profile['pos_ADJ_mean'])}.\n"
        "- استخدم صفات لوصف الأشخاص والأشياء والأفكار.\n"
        "- كلما زاد المستوى، زادت التفاصيل.\n\n"
        f"4) بناء الجملة والمعنى:\n"
        f"- عدد الفاعل (قريب من {round(profile['dep_SBJ_mean'])}).\n"
        f"- عدد مفاعيل وأمثلة (قريب من {round(profile['dep_OBJ_mean'])}).\n"
        "- في المستويات المتقدمة(B1-C2)، اربط السبب بالنتيجة.\n\n"
        f"5) البنية النحوية (شجرة التحليل):\n"
        f"- العمق الأدنى لشجرة التحليل قريب من {depth_min}.\n"
        f"- العمق الأقصى لشجرة التحليل قريب من {depth_max}.\n"
        "- كلما ارتفع المستوى، استخدم جملًا ذات بنية أعمق وأكثر تفرعًا.\n\n"
        "6) الترابط والمنطق:\n"
        "- استخدم روابط منطقية تناسب المستوى.\n"
        "- في المستويات المتقدمة، استخدم التعليل والمقارنة.\n\n"
        "7) الأسلوب العام:\n"
        "- لا أسلوب حواري إلا في B2 أو أعلى.\n"
        "- تجنّب التعجب والمبالغة في المستويات الدنيا.\n"
        "- الأسلوب يجب أن يعكس مستوى متعلم حقيقي.\n\n"
        "قيود مهمة:\n"
        "- هذه القيم إرشادية فقط.\n"
        "- لا تذكر أي أرقام في النص.\n"
        "- لا تشرح هذه القيود في الناتج.\n\n"
        "هيكل النص:\n"
        "- مقدمة، محتوى، خاتمة بدون عناوين.\n"
        "- فقرات متصلة فقط.\n"
    )


PROMPT_BUILDERS = {
    "P1": build_p1,
    "P2": build_p2,
    "P3": build_p3,
    "P4": build_p4,
    "P5": build_p5,
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Build the batch JSONL file.

    Args:
        args: Parsed CLI arguments.

    Raises:
        ValueError: If required inputs are missing for the chosen prompt version.
    """
    prompt_version = args.prompt_version.upper()
    if prompt_version not in PROMPT_BUILDERS:
        raise ValueError(f"--prompt_version must be one of {list(PROMPT_BUILDERS)}.")

    # Validate required inputs per prompt version
    if prompt_version in ("P3", "P5") and not args.profile_csv:
        raise ValueError(f"{prompt_version} requires --profile_csv.")
    if prompt_version in ("P4", "P5") and not args.vocab_csv:
        raise ValueError(f"{prompt_version} requires --vocab_csv.")

    # --- Load profiles ---
    profiles: Dict[str, dict] = {}
    if args.profile_csv:
        profiles = load_profiles(args.profile_csv)
        print(f"[build_batch] Loaded profiles for levels: {list(profiles.keys())}")

    # --- Load vocab ---
    vocab_map: Dict[Tuple[str, str], list] = {}
    if args.vocab_csv:
        vocab_map = load_vocab_map(args.vocab_csv)
        print(f"[build_batch] Loaded vocab for {len(vocab_map)} prompt/level pairs.")

    # --- Process prompts ---
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    skipped  = 0
    written  = 0
    builder  = PROMPT_BUILDERS[prompt_version]

    with open(args.prompts_csv, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl,   "w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        for line_num, row in enumerate(reader, start=1):
            prompt_id  = row.get("prompt_id",  "").strip()
            level      = row.get("CEFR_level", "").strip()
            topic_text = row.get("Arabic Text", "").strip()

            if not prompt_id or not level or not topic_text:
                print(
                    f"[WARNING] Line {line_num}: missing prompt_id, CEFR_level, "
                    "or Arabic Text. Skipping."
                )
                skipped += 1
                continue

            if level not in TREE_DEPTH_MAP:
                print(f"[WARNING] Line {line_num}: unknown CEFR level '{level}'. Skipping.")
                skipped += 1
                continue

            depth_min, depth_max = TREE_DEPTH_MAP[level]

            # Resolve profile
            profile = profiles.get(level, {})
            if prompt_version in ("P3", "P5") and not profile:
                print(
                    f"[WARNING] Line {line_num}: no profile for level '{level}'. Skipping."
                )
                skipped += 1
                continue

            # Resolve vocab
            words: list = []
            if prompt_version in ("P4", "P5"):
                words = vocab_map.get((prompt_id, level), [])
                if not words:
                    print(
                        f"[WARNING] prompt_id={prompt_id} level={level}: "
                        "no vocab found. Proceeding with empty word list."
                    )

            user_prompt = builder(
                topic_text=topic_text,
                level=level,
                profile=profile,
                depth_min=depth_min,
                depth_max=depth_max,
                words=words,
            )

            custom_id  = f"{prompt_id}_{level}_{args.custom_id_suffix}"
            batch_line = {
                "custom_id": custom_id,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature": args.temperature,
                },
            }
            fout.write(json.dumps(batch_line, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[build_batch] Done. Written: {written} | Skipped: {skipped} | "
        f"Output: {args.out_jsonl}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a GPT batch input JSONL for Arabic essay generation."
    )
    p.add_argument(
        "--prompts_csv",
        required=True,
        help="Input CSV with columns: prompt_id, CEFR_level, Arabic Text.",
    )
    p.add_argument(
        "--out_jsonl",
        required=True,
        help="Output JSONL path for OpenAI batch API.",
    )
    p.add_argument(
        "--prompt_version",
        required=True,
        choices=["P1", "P2", "P3", "P4", "P5"],
        help=(
            "Prompting condition: "
            "P1=no level/vocab/profile, "
            "P2=level only, "
            "P3=level+profile, "
            "P4=level+vocab, "
            "P5=level+profile+vocab."
        ),
    )
    p.add_argument(
        "--profile_csv",
        default=None,
        help="Level profile CSV (required for P3 and P5). Must contain a 'level' column.",
    )
    p.add_argument(
        "--vocab_csv",
        default=None,
        help="Vocab CSV (required for P4 and P5). Must contain: prompt_id, CEFR_level, words.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    p.add_argument(
        "--custom_id_suffix",
        default="p1",
        help="Suffix for custom_id field in each batch line (default: p1).",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)