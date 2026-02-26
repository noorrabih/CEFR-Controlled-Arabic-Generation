import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conllx_df.src.conllx_df import ConllxDf
from conllx_df.src.conll_utils import get_token_details, get_sentence_column_data, get_children_ids_of, get_parent_id, add_parent_details, add_direction, get_token_count
import os
import pandas as pd
import tqdm


src_dir = "/home/nour.rabih/arwi/readability_controlled_generation/generation/no_level/essays_parsed_Text_100"
stats_dir = os.path.join(src_dir, "tree_stats")
os.makedirs(stats_dir, exist_ok=True)

def extract_sentence_ids(conll_file):
    """Return the list of sentence IDs from '# id = ...' lines in correct order."""
    sent_ids = []
    with open(conll_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# id ="):
                sent_ids.append(line.split("=", 1)[1].strip())
    return sent_ids


def calculate_tree_stats(sent_df):
    """
    Calculate depth and branching factor statistics for a dependency tree.
    
    Args:
        sent_df: DataFrame containing sentence data
        
    Returns:
        dict: Dictionary containing tree statistics
    """
    # Build the tree structure
    tree = [[0]]
    while True:
        new_level = []
        for parent_id in tree[-1]:
            children_ids = get_children_ids_of(sent_df, parent_id)
            new_level.extend(children_ids)
        if len(new_level) == 0:
            break
        tree.append(new_level)
    
    # Calculate depth (excluding root level)
    depth = len(tree) - 1
    breadth = max([len(level) for level in tree])
    
    # Calculate branching factors for each level
    branching_factors = []
    for level in tree[1:]:
        level_branching = []
        for parent_id in level:
            children_count = len(get_children_ids_of(sent_df, parent_id))
            if children_count > 0:  # Only count nodes that have children
                level_branching.append(children_count)
        if level_branching:
            branching_factors.extend(level_branching)
    
    # Calculate statistics
    total_nodes = get_token_count(sent_df)
    stats = {
        'depth': depth,
        'breadth': breadth,
        'depth_normalized': depth / total_nodes if total_nodes > 0 else 0,
        'breadth_normalized': breadth / total_nodes if total_nodes > 0 else 0,
        'max_branching_factor': max(branching_factors) if branching_factors else 0,
        'avg_branching_factor': sum(branching_factors) / len(branching_factors) if branching_factors else 0,
        'min_branching_factor': min(branching_factors) if branching_factors else 0,
        'total_nodes': total_nodes,
        'tree_structure': tree,  # Include all levels
        'branching_factors': branching_factors
    }
    
    return stats

files = [f for f in os.listdir(src_dir) if f.endswith(".conllx")]
files_stats = {
    'file_name': [],
    'total_sentences': [],
    'total_nodes': [],
    'avg_depth': [],
    'avg_breadth': [],
    'avg_depth_normalized': [],
    'avg_breadth_normalized': [],
    'avg_branching_factor': []
}


for file in tqdm.tqdm(files):
    conll_file = os.path.join(src_dir, file)
    sentence_ids = extract_sentence_ids(conll_file)
    with open(conll_file, 'r', encoding='utf-8') as f:
        data = f.read().split("\n")
    sent_count = len([line for line in data if line.startswith("# id =")])
    conll_df = ConllxDf(conll_file)
    stats_df_dict = {
        'sentence_id': [],
        'depth': [],
        'breadth': [],
        'depth_normalized': [],
        'breadth_normalized': [],
        'max_branching_factor': [],
        'avg_branching_factor': [],
        'min_branching_factor': [],
        'total_nodes': [],
        'branching_factors': []
    }
    for i, sent_id in enumerate(sentence_ids):
        sent_df = conll_df.get_df_by_id(i)
        stats = calculate_tree_stats(sent_df)
        stats_df_dict['sentence_id'].append(sent_id)
        stats_df_dict['depth'].append(stats['depth'])
        stats_df_dict['breadth'].append(stats['breadth'])
        stats_df_dict['depth_normalized'].append(stats['depth_normalized'])
        stats_df_dict['breadth_normalized'].append(stats['breadth_normalized'])
        stats_df_dict['max_branching_factor'].append(stats['max_branching_factor'])
        stats_df_dict['avg_branching_factor'].append(stats['avg_branching_factor'])
        stats_df_dict['min_branching_factor'].append(stats['min_branching_factor'])
        stats_df_dict['total_nodes'].append(stats['total_nodes'])
        stats_df_dict['branching_factors'].extend(stats['branching_factors'])
    branching_factors_list = stats_df_dict.pop('branching_factors')
    stats_df = pd.DataFrame.from_dict(stats_df_dict)
    stats_df.to_excel(os.path.join(stats_dir, file.replace(".conllx", ".xlsx")), index=False)
    files_stats['file_name'].append(file)
    files_stats['total_sentences'].append(sent_count)
    files_stats['total_nodes'].append(sum(stats_df_dict['total_nodes']))
    files_stats['avg_depth'].append(sum(stats_df_dict['depth'])/len(stats_df_dict['depth']))
    files_stats['avg_breadth'].append(sum(stats_df_dict['breadth'])/len(stats_df_dict['breadth']))
    files_stats['avg_depth_normalized'].append(sum(stats_df_dict['depth_normalized'])/len(stats_df_dict['depth_normalized']))
    files_stats['avg_breadth_normalized'].append(sum(stats_df_dict['breadth_normalized'])/len(stats_df_dict['breadth_normalized']))
    files_stats['avg_branching_factor'].append(sum(branching_factors_list)/len(branching_factors_list))
files_stats_df = pd.DataFrame.from_dict(files_stats)
files_stats_df.to_excel(os.path.join(stats_dir, "all_files_stats.xlsx"), index=False)
