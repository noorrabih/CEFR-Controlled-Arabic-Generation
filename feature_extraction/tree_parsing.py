"""
Arabic Sentence Parsing and CoNLL-X Export for Syntactic Analysis

This script batches and parses a dataset of Arabic sentences using the CAMeL Parser .
 It performs the following:

1. Splits input data into CSV batches.
2. Parses each batch into dependency trees using the CAMeL parser.
3. Saves the parsed output in CoNLL-X format with sentence IDs for traceability.

Input:  Excel file with 'ID' and 'Text' columns.
Output: CoNLL-X files (one per batch) for syntactic feature extraction.

Folders:
- Input batches:     batches_Text_100
- Output parsed files: parsed_batches_100
"""
import sys
import os
# sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import pandas as pd
from pathlib import Path
from camel_tools.utils.charmap import CharMapper
from pandas import read_csv
from camel_parser.src.data_preparation import get_tagset, parse_text
from camel_parser.src.initialize_disambiguator.disambiguator_interface import get_disambiguator
from camel_parser.src.classes import TextParams
from camel_parser.src.conll_output import text_tuples_to_string
from typing import List
import os
import pandas as pd



# data_file_path =  "../../thesis_data/1M_features/All_data_1M_morph_clean.csv"
# data = pd.read_csv(data_file_path)

def batch_data(data_df, batch_output_dir):
    """
    Splits the input DataFrame into batches and saves each batch as a CSV file.
    Args:
        data_df: Input DataFrame with 'ID' and 'Clean_Sentnece' columns.
        batch_output_dir: Directory to save the batch CSV files.
    """
    # Define output directories for batches and parsed data
    # parsed_output_dir = "1M_features/parsed_batches_100"
    os.makedirs(batch_output_dir, exist_ok=True)

    # Batch size
    batch_size = 100
    i = 0
    # Splitting the data into batches
    for batch_number, start_idx in enumerate(range(0, len(data_df), batch_size), start=1):
        batch = data_df.iloc[start_idx:start_idx + batch_size][["ID", "Sentence"]]
        batch_file_path = os.path.join(batch_output_dir, f"batch_{batch_number}.csv")
        batch.to_csv(batch_file_path, index=False)


    # Output result
    f"Data has been batched into {batch_number} files and saved to '{batch_output_dir}'."

# Initialize model parameters
model_path = Path("/home/nour.rabih/Readability-morph/camel_parser/models")
parse_model = "catib"
arclean = CharMapper.builtin_mapper("arclean")
clitic_feats_df = read_csv('/home/nour.rabih/camel_parser/data/clitic_feats.csv')
clitic_feats_df = clitic_feats_df.astype(str).astype(object)
model_name = "CAMeLBERT-CATiB-biaffine.model"
tagset = get_tagset(parse_model)
disambiguator = get_disambiguator("bert", "calima-msa-s31")


# Folder containing CSV batch files
# input_folder = "1M_features/batches_Text_100"
# output_folder = "1M_features/parsed_batches_100"


def parse_data(input_folder, output_folder):
    """
    Parses all CSV files in the input folder and saves the parsed output in CoNLL-X format.
    Args:
        input_folder: Directory containing the CSV files to parse.
        output_folder: Directory to save the parsed CoNLL-X files.
    """
    os.makedirs(output_folder, exist_ok=True)
    # Loop over CSV batches and parse them
    for batch_file in os.listdir(input_folder):
        # if batch_file not already in output_folder
        if os.path.exists(os.path.join(output_folder, batch_file.replace(".csv", ".conllx"))):
            print(f"Skipping {batch_file} as it has already been parsed.")
        else:
            if batch_file.endswith(".csv") :
                # Load CSV
                batch_path = os.path.join(input_folder, batch_file)
                batch_df = pd.read_csv(batch_path)

                # Ensure it has 'id' and 'sentence' columns
                if 'ID' not in batch_df.columns or 'Sentence' not in batch_df.columns:
                    raise ValueError(f"CSV {batch_file} must contain 'ID' and 'Sentence' columns.")

                # Extract sentences and IDs
                sentences = batch_df['Sentence'].tolist()
                sentence_ids = batch_df['ID'].tolist()

                # Pass sentences and parameters to TextParams
                file_type_params = TextParams(sentences, model_path / model_name, arclean, disambiguator, clitic_feats_df, tagset, "")
                parsed_text_tuples = parse_text("text", file_type_params)

                # Convert to CONLLX format string
                trees_string = text_tuples_to_string(parsed_text_tuples, file_type='conll', sentences=sentences) 

                # join trees_string to one big string
                trees_string = "\n".join(trees_string)
                # Save to a .conllx file
                output_path = os.path.join(output_folder, batch_file.replace(".csv", ".conllx"))
                with open(output_path, 'w', encoding='utf-8') as conllx_file:
                    # Append sentence IDs as comments for traceability
                    for i, tree in enumerate(trees_string.split('\n\n')):
                        conllx_file.write(f"# id = {sentence_ids[i]}\n")
                        conllx_file.write(tree + "\n\n")


# if __name__ == "__main__":
#     input_file = "../data/All_data_1M_morph_clean.csv"
#     df = pd.read_csv(input_file)[:100]
#     batch_output_dir = "../data/batches_Text_100"
#     batch_data(df, batch_output_dir)
#     output_folder = '"../data/batches_parsed_100'
#     # Parse all sentences
#     parsed_trees = parse_data(batch_output_dir, output_folder)

#     print(f"✅ Done. Parsed trees saved ")