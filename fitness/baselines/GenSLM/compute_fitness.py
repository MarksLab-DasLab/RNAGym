import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from genslm import GenSLM, SequenceDataset


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


def process_single_row(row, model, device, base_dir, results_dir, score_column):
    dataset = row['fitness filename']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = row['Raw Construct Sequence'].upper()
    try:
        sequences = get_sequences(wt_seq, df)
    except AssertionError as e:
        print("assertion error", e, "in", dataset)
        return

    output_file = os.path.join(results_dir, f"{dataset}.csv")

    seq_length = min(model.seq_length, sequences["mutated_sequence"].str.len().max() + 2)
    if model.seq_length < sequences["mutated_sequence"].str.len().max() + 2:
        print("warning: max str length exceeded")
    dataset = SequenceDataset(sequences["mutated_sequence"], seq_length, model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=4)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses = []
    lengths = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
            loss = (
                loss_fn(outputs.logits.cpu().permute((0, 2, 1)), batch["input_ids"]) *
                batch["attention_mask"].squeeze(1)
            ).sum(1)
            losses.append(loss)
            lengths.append(batch["attention_mask"].squeeze(1).sum(1))
    losses = np.concatenate(losses)
    lengths = np.concatenate(lengths)

    sequences[score_column] = losses
    sequences.to_csv(output_file, index=False)


def main(args):
    model = GenSLM("genslm_2.5B_patric", model_cache_dir=os.path.join(args.checkpoint_dir, "2.5B"))
    model.eval()

    # Select GPU device if it is available, else use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    base_dir = args.dms_directory
    results_dir = args.output_directory
    score_column = 'logit_scores'
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs.dropna(inplace=True)
    wt_seqs['Year'] = wt_seqs['Year'].astype(int)
    wt_seqs['First Author Last Name'] = wt_seqs['First Author Last Name'].astype(str)
    wt_seqs['Molecule Type'] = wt_seqs['Molecule Type'].astype(str)

    # Select the row corresponding to the task ID
    row = wt_seqs.iloc[args.task_id]

    process_single_row(row, model, device, base_dir, results_dir, score_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument("--checkpoint_dir", type=str, help="GenSLM checkpoints directory")
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")
    args = parser.parse_args()
    main(args)