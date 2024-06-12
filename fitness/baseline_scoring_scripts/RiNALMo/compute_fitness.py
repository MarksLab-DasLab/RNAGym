import torch
import pandas as pd
from rinalmo.pretrained import get_pretrained_model
import os
import argparse
from scipy import stats
import numpy as np

def get_sequences(wt_sequence, df):
    def apply_mutation(sequence, mutation):
        sequence = sequence.replace('T', 'U')
        possible_bases = ['A', 'U', 'C', 'G', 'N', '']
        mutation = mutation.replace(' ', '')
        pos = int(mutation[1:-1]) - 1
        new_base = mutation[-1]
        old_base = mutation[0]
        old_base = 'U' if old_base == 'T' else old_base
        new_base = 'U' if new_base == 'T' else new_base

        assert old_base in possible_bases, mutation
        assert new_base in possible_bases, mutation

        if old_base != 'N':  
            assert old_base == sequence[pos], mutation
        
        if new_base == '':
            
            mutated_sequence = sequence[:pos] + sequence[pos+1:]
        else:
            
            mutated_sequence = sequence[:pos] + new_base + sequence[pos+1:]

        return mutated_sequence

    def apply_mutations(sequence, mutations):
        for mutation in mutations.split(','):
            sequence = apply_mutation(sequence, mutation)
        return sequence

    mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None
    if mutation_column:
        df['mutated_sequence'] = df[mutation_column].apply(lambda x: apply_mutations(wt_sequence, x))
    else:
        raise ValueError("No 'mutant' or 'mutation' column found in the DataFrame")
    
    return df

def apply_masked_marginal_scoring(wt_sequence, mutated_sequence, positions, model, alphabet, device):
    
    tokens = torch.tensor(alphabet.batch_tokenize([mutated_sequence]), dtype=torch.int64).to(device)
    total_score = 0
    possible_bases = ['A', 'U', 'C', 'G']

    for pos in positions:
        
        masked_tokens = tokens.clone()
        masked_tokens[0, pos + 1] = alphabet.mask_idx 
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            
            results = model(masked_tokens)
        
        # Calculate log probabilities
        token_logits = results["logits"]
        token_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        
        # Get log probability of the mutated base at the position
        mut_encoded = alphabet.get_idx(mutated_sequence[pos])
        log_prob_mut = token_probs[0, pos + 1, mut_encoded].item()
    
        #remove N's from WT sequence 
        wt_sequence = wt_sequence.replace('N', '')
        
        wt_encoded = alphabet.get_idx(wt_sequence[pos])
        log_prob_wt = token_probs[0, pos + 1, wt_encoded].item()
        
        # Calculate the difference and add to the total score
        total_score += log_prob_mut - log_prob_wt

    return total_score

def extract_positions(mutation, offset=1):
    positions = []
    for mut in mutation.split(','):
        pos = int(mut[1:-1]) - offset
        positions.append(pos)
    return positions

def process_single_row(row, model, alphabet, device, base_dir,results_dir, score_column):
    dataset = row['fitness filename']
    if 'snoRNA' in dataset:
        return
    df_path = os.path.join(base_dir, f'{dataset}.csv')
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.lower()
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    wt_seq = row['Raw Construct Sequence (DNA)'].upper()
    sequences = get_sequences(wt_seq, df)

    logit_scores = []
    for _, seq in sequences.iterrows():
        mutation_column = 'mutant' if 'mutant' in sequences.columns else 'mutation' if 'mutation' in sequences.columns else 'mutations' if 'mutations' in sequences.columns else None
        positions = extract_positions(seq[mutation_column])
        logits = apply_masked_marginal_scoring(wt_seq,seq['mutated_sequence'], positions, model, alphabet, device)
        logit_scores.append(logits)

    sequences[score_column] = logit_scores
    output_file = os.path.join(results_dir, f"{dataset}.csv")
    sequences.to_csv(output_file, index=False)


def main(args):
    
    DEVICE = "cuda:0"
    model, alphabet = get_pretrained_model(model_name="giga-v1")
    model = model.to(device=DEVICE)
    model.eval()

    base_dir = 'processed/'
    results_dir = 'results'
    score_column = 'logit_scores'
    wt_seqs = pd.read_csv(args.reference_sheet, encoding='latin-1')
    wt_seqs.dropna(inplace=True)
    wt_seqs['Year'] = wt_seqs['Year'].astype(int)
    wt_seqs['First Author Last Name'] = wt_seqs['First Author Last Name'].astype(str)
    wt_seqs['Molecule Type'] = wt_seqs['Molecule Type'].astype(str)

    # Select the row corresponding to the task ID
    row = wt_seqs.iloc[args.task_id]

    process_single_row(row, model, alphabet, DEVICE, base_dir,results_dir, score_column)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sheet', type=str, required=True)
    parser.add_argument('--task_id', type=int, required=True)
    args = parser.parse_args()
    main(args)