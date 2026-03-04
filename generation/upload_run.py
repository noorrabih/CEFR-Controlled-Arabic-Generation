"""Upload a Batch JSONL file to OpenAI and (optionally) retrieve results.

This script uses the OpenAI Python SDK (>=1.x).

Typical workflow:
1) Create batch_input.jsonl with generation/build_batch.py
2) Upload + create batch:
   python generation/upload_run.py --batch_input batch_input.jsonl --out_tsv generated_essays.tsv

Notes:
- Requires OPENAI_API_KEY in environment.
- By default, this script will create the batch and then poll until completion.
  Use --no_poll if you want to stop after creating the batch.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
from openai import OpenAI


def extract_essay_from_response_line(line: str) -> tuple[str, str] | None:
    """Return (custom_id, essay_text) from a batch output line."""
    try:
        obj = json.loads(line)
        custom_id = obj.get("custom_id")
        body = obj.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "")
        return custom_id, content
    except Exception:
        return None


def main(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    # Upload file
    with open(args.batch_input, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"description": args.description} if args.description else None,
    )

    print("Batch created:", batch.id)

    if args.no_poll:
        return

    # Poll
    while True:
        b = client.batches.retrieve(batch.id)
        status = b.status
        print("Status:", status)
        if status in {"completed", "failed", "cancelled", "expired"}:
            batch = b
            break
        time.sleep(args.poll_seconds)

    if batch.status != "completed":
        raise RuntimeError(f"Batch did not complete successfully. Status: {batch.status}")

    if not batch.output_file_id:
        raise RuntimeError("Batch completed but no output_file_id found.")

    # Download output file content
    content = client.files.content(batch.output_file_id).text

    rows = []
    for line in content.splitlines():
        parsed = extract_essay_from_response_line(line)
        if parsed:
            custom_id, essay_text = parsed
            rows.append({"ID": custom_id, "essay": essay_text})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote: {args.out_tsv} ({len(out_df)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Upload and run an OpenAI Batch job for essay generation.")
    ap.add_argument("--batch_input", required=True, help="Path to batch_input.jsonl")
    ap.add_argument("--out_tsv", required=True, help="Output TSV for generated essays")
    ap.add_argument("--completion_window", default="24h")
    ap.add_argument("--description", default=None)
    ap.add_argument("--poll_seconds", type=int, default=10)
    ap.add_argument("--no_poll", action="store_true")
    main(ap.parse_args())
