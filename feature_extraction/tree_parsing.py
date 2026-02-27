"""Batch and parse Arabic sentences into CoNLL-X using CAMeL Parser.

Requirements:
- camel-tools
- camel-parser (the CAMeL Parser repo/package must be importable)

This script:
1) Reads an input CSV with columns: ID, Sentence
2) Splits into batches (default 100 rows per batch) and writes batch_*.csv
3) Parses each batch and writes batch_*.conllx with '# id = <sentence_id>' comments

Example:
  python feature_extraction/tree_parsing.py \
    --input_csv generation/.../generated_essays_sentences.csv \
    --batch_dir essays_batches_Text_100 \
    --out_dir essays_parsed_Text_100 \
    --batch_size 100 \
    --model_dir /path/to/camel_parser/models \
    --clitic_feats_csv /path/to/camel_parser/data/clitic_feats.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def batch_data(data_df: pd.DataFrame, batch_output_dir: str, batch_size: int = 100) -> int:
    os.makedirs(batch_output_dir, exist_ok=True)

    n_batches = 0
    for batch_number, start_idx in enumerate(range(0, len(data_df), batch_size), start=1):
        batch = data_df.iloc[start_idx:start_idx + batch_size][["ID", "Sentence"]]
        batch_file_path = os.path.join(batch_output_dir, f"batch_{batch_number}.csv")
        batch.to_csv(batch_file_path, index=False)
        n_batches = batch_number

    return n_batches


def _init_parser(model_dir: str, clitic_feats_csv: str, parse_model: str = "catib", model_name: str = "CAMeLBERT-CATiB-biaffine.model"):
    """Initialize CAMeL Parser dependencies lazily (so imports don't fail unless used)."""
    # Ensure the directory that contains the camel_parser package is on sys.path.
    # model_dir is expected to be <root>/camel_parser/models, so two levels up is <root>.
    camel_parser_root = str(Path(model_dir).parent.parent)
    if camel_parser_root not in sys.path:
        sys.path.insert(0, camel_parser_root)

    from camel_tools.utils.charmap import CharMapper
    from pandas import read_csv

    from camel_parser.src.data_preparation import get_tagset, parse_text
    from camel_parser.src.initialize_disambiguator.disambiguator_interface import get_disambiguator
    from camel_parser.src.classes import TextParams
    from camel_parser.src.conll_output import text_tuples_to_string

    model_path = Path(model_dir)
    arclean = CharMapper.builtin_mapper("arclean")
    clitic_feats_df = read_csv(clitic_feats_csv).astype(str).astype(object)
    tagset = get_tagset(parse_model)
    disambiguator = get_disambiguator("bert", "calima-msa-s31")

    return {
        "parse_text": parse_text,
        "TextParams": TextParams,
        "text_tuples_to_string": text_tuples_to_string,
        "model_path": model_path,
        "model_name": model_name,
        "arclean": arclean,
        "disambiguator": disambiguator,
        "clitic_feats_df": clitic_feats_df,
        "tagset": tagset,
    }


def parse_batches(input_folder: str, output_folder: str, parser_cfg: dict) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for batch_file in sorted(os.listdir(input_folder)):
        if not batch_file.endswith(".csv"):
            continue

        out_path = os.path.join(output_folder, batch_file.replace(".csv", ".conllx"))
        if os.path.exists(out_path):
            print(f"Skipping {batch_file} (already parsed)")
            continue

        batch_df = pd.read_csv(os.path.join(input_folder, batch_file))
        if 'ID' not in batch_df.columns or 'Sentence' not in batch_df.columns:
            raise ValueError(f"CSV {batch_file} must contain 'ID' and 'Sentence' columns.")

        sentences = batch_df['Sentence'].tolist()
        sentence_ids = batch_df['ID'].tolist()

        file_type_params = parser_cfg["TextParams"](
            sentences,
            parser_cfg["model_path"] / parser_cfg["model_name"],
            parser_cfg["arclean"],
            parser_cfg["disambiguator"],
            parser_cfg["clitic_feats_df"],
            parser_cfg["tagset"],
            "",
        )

        parsed_text_tuples = parser_cfg["parse_text"]("text", file_type_params)
        trees_string_list = parser_cfg["text_tuples_to_string"](parsed_text_tuples, file_type='conll', sentences=sentences)
        trees_string = "\n".join(trees_string_list)

        with open(out_path, 'w', encoding='utf-8') as conllx_file:
            for i, tree in enumerate(trees_string.split('\n\n')):
                conllx_file.write(f"# id = {sentence_ids[i]}\n")
                conllx_file.write(tree + "\n\n")

        print(f"Wrote: {out_path}")


def main(args):
    df = pd.read_csv(args.input_csv)
    if "ID" not in df.columns or "Sentence" not in df.columns:
        raise ValueError("Input CSV must contain columns: ID, Sentence")

    n_batches = batch_data(df, args.batch_dir, batch_size=args.batch_size)
    print(f"Batched {len(df)} rows into {n_batches} files at: {args.batch_dir}")

    parser_cfg = _init_parser(args.model_dir, args.clitic_feats_csv, parse_model=args.parse_model, model_name=args.model_name)
    parse_batches(args.batch_dir, args.out_dir, parser_cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch and parse Arabic sentences into CoNLL-X using CAMeL Parser.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=100)

    ap.add_argument("--model_dir", default="/home/nour.rabih/Readability-morph/camel_parser/models", help="Path to camel_parser/models directory")
    ap.add_argument("--clitic_feats_csv", default="/home/nour.rabih/camel_parser/data/clitic_feats.csv", help="Path to camel_parser/data/clitic_feats.csv")
    ap.add_argument("--parse_model", default="catib")
    ap.add_argument("--model_name", default="CAMeLBERT-CATiB-biaffine.model")

    main(ap.parse_args())


