from transformers import AutoTokenizer, AutoModelForMaskedLM
from numpy import dot
from numpy.linalg import norm
from Bio import SeqIO
import torch
import argparse
import pandas as pd
import os
import copy
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
import sys


def get_sequences(wt_sequence, df):
    
    # Function to apply a single mutation
    def apply_mutation(sequence, mutation):
        sequence= sequence.replace('U', 'T')
        possible_bases = ['A', 'T', 'C', 'G', 'N', '']
        mutation = mutation.replace(' ', '')
        pos = int(mutation[1:-1]) - 1  # Get the position, 1-based to 0-based index
        new_base = mutation[-1]  # Get the new base
        old_base = mutation[0]
        old_base = 'T' if old_base == 'U' else old_base
        new_base = 'T' if new_base == 'U' else new_base

        assert old_base in possible_bases, mutation
        assert new_base in possible_bases, mutation

        if old_base == 'N':
            # This is going to be an insertion
            mutated_sequence = sequence[:pos+1] + new_base + sequence[pos+1:]
        elif new_base == '':
            # This is going to be a deletion
            mutated_sequence = sequence[:pos] + sequence[pos+1:]
        else:
            # This is a substitution
            assert old_base == sequence[pos], mutation
            mutated_sequence = sequence[:pos] + new_base + sequence[pos+1:]

        return mutated_sequence

    # Function to apply multiple mutations
    def apply_mutations(sequence, mutations):
        for mutation in mutations.split(','):
            sequence = apply_mutation(sequence, mutation)
        return sequence

    # Determine which column to use for mutations
    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None
    if mutation_column:
        # Apply the mutations to create a new column with mutated sequences
        df['mutated_sequence'] = df[mutation_column].apply(lambda x: apply_mutations(wt_sequence, x))
    else:
        raise ValueError("No 'mutant' or 'mutation' column found in the DataFrame")

    return df

def score_variants(assay, model, tokenizer, base_dir, results_dir, score_column, batch_size=32):
    
    dataset = assay['fitness filename']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = assay['Raw Construct Sequence'].upper()
    
    try:
        sequences = get_sequences(wt_seq, df)
    except AssertionError as e:
        print("assertion error", e, "in", dataset)
        return
    
    output_file = os.path.join(results_dir, f"{dataset}.csv")

    max_length = tokenizer.model_max_length
    scores = []

    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch = sequences[i:i + batch_size]

        # Check and truncate sequences if they exceed the model's maximum length
        batch = [seq[:max_length] for seq in batch]

        # Tokenize the batch of sequences
        tokens = tokenizer(batch, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True).to(device)

        # Process the batch
        with torch.no_grad():
            logits = model(input_ids=tokens['input_ids']).logits
            normalized_logits = F.log_softmax(logits, dim=-1)

        # Compute average log probability of the correct tokens
        for j, seq in enumerate(batch):
            input_ids = tokens['input_ids'][j]
            # Ignore padding tokens
            valid_length = (input_ids != tokenizer.pad_token_id).sum().item()

            if valid_length == 0:
                continue

            # Get log probabilities of the correct tokens
            log_probs = normalized_logits[j, torch.arange(valid_length), input_ids[:valid_length]]
            avg_log_prob = log_probs.mean().item()

            scores.append(avg_log_prob)

    scores.to_csv(f'{output_file}')


def main(args):

    print('Loading model...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_location)
    model = AutoModelForMaskedLM.from_pretrained(args.model_location).to(device)

    base_dir = args.dms_directory
    results_dir = args.output_directory
    score_column = 'logit_scores'
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs.dropna(inplace=True)
    wt_seqs['Year'] = wt_seqs['Year'].astype(int)
    wt_seqs['First Author Last Name'] = wt_seqs['First Author Last Name'].astype(str)
    wt_seqs['Molecule Type'] = wt_seqs['Molecule Type'].astype(str)

    # Select the row corresponding to the task ID
    assay = wt_seqs.iloc[args.task_id]
    score_variants(assay, model, tokenizer, base_dir, results_dir, score_column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument("--model_location", type=str, help="NT Huggingface model to use")
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")
    args = parser.parse_args()
    main(args)


