import os
import pandas as pd
import numpy as np
from Bio import pairwise2
from sklearn.model_selection import train_test_split
import tqdm

def load_dms_data(dms_location, reference_file, assay_filename_col = "DMS_ID"):
    """Load all DMS data from CSV files."""
    reference_df = pd.read_csv(reference_file)
    all_data = {}
    
    for filename in reference_df[assay_filename_col]:
        file_path = os.path.join(dms_location, filename + '.csv')
        df = pd.read_csv(file_path)
        all_data[filename] = df
    
    return all_data

def random_split_assay(assay_data, test_size=0.2, random_state=42):
    """Perform random train-validation split for a single assay."""
    return train_test_split(assay_data, test_size=test_size, random_state=random_state)

def sequence_similarity(seq1, seq2):
    """Calculate sequence similarity between two sequences."""
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    return alignment.score / max(len(seq1), len(seq2))

def minimize_similarity_split_assay(assay_data, test_size=0.2, max_iterations=10000):
    """Perform train-validation split minimizing sequence similarity for a single assay."""
    train, val = random_split_assay(assay_data, test_size)
    
    for _ in tqdm.tqdm(range(max_iterations)):
        improved = False
        for i, row_val in val.iterrows():
            for j, row_train in train.iterrows():
                if sequence_similarity(row_val['sequence'], row_train['sequence']) > 0.8:  # Threshold
                    val = pd.concat([val.drop(i), pd.DataFrame([row_train])], ignore_index=True)
                    train = pd.concat([train.drop(j), pd.DataFrame([row_val])], ignore_index=True)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return train, val

def save_splits(splits, output_dir, split_type):
    """Save train and validation splits to CSV files, one file per assay."""
    split_dir = os.path.join(output_dir, split_type)
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for assay, (train, val) in splits.items():
        train_file = os.path.join(train_dir, f"{assay.replace('.csv', '')}_train.csv")
        val_file = os.path.join(val_dir, f"{assay.replace('.csv', '')}_val.csv")
        
        train.to_csv(train_file, index=False)
        val.to_csv(val_file, index=False)
    
    print(f"Saved {split_type} split files:")
    print(f"  Train directory: {train_dir}")
    print(f"  Validation directory: {val_dir}")

def main(dms_location, reference_file, output_dir):
    # Load data
    data = load_dms_data(dms_location, reference_file)
    
    # Random split for each assay
    random_splits = {}
    for assay, assay_data in data.items():
        train_random, val_random = random_split_assay(assay_data)
        random_splits[assay] = (train_random, val_random)
        print(f"Random split for {assay} - Train size: {len(train_random)}, Validation size: {len(val_random)}")
    save_splits(random_splits, output_dir, "random")
    
    # Minimize similarity split for each assay
    sim_splits = {}
    for assay, assay_data in data.items():
        train_sim, val_sim = minimize_similarity_split_assay(assay_data)
        sim_splits[assay] = (train_sim, val_sim)
        print(f"Similarity-based split for {assay} - Train size: {len(train_sim)}, Validation size: {len(val_sim)}")
    save_splits(sim_splits, output_dir, "min_similarity")

if __name__ == "__main__":
    DMS_location = "Path to preprocessed DMS files"  # Replace with actual path
    reference_file = "../reference_sheet.csv"  # Replace with actual path
    output_dir = "./fitness_CV_splits"  # Replace with desired output directory
    main(DMS_location, reference_file, output_dir)