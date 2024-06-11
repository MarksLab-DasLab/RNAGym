import os
import torch
import sys
import numpy as np
from scipy import stats
import argparse
import matplotlib.pyplot as plt
import pandas as pd


def create_parser(): 
    parser = argparse.ArgumentParser(description='Label a RNA mutation dataset with zero-shot wild-type marginal predictions from RNA-FM')
    parser.add_argument("--model_location", type=str, help="RNA-FM model directory")
    parser.add_argument("--reference_sequences", type=str, help="CSV file containing the reference sequences")
    parser.add_argument("--dms_directory", type=str,help="Directory containing the mutational datasets to be scored")
    parser.add_argument("--output_directory", type=str, help="Directory for scored fitness files")        
    return parser       


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    score=0
    for mutation in row.split(","): # change split to comma
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
    return score


def create_dms_file_path(filename):
	return f'{args.dms_directory}{filename}.csv'


def main(args):
	print("Arguments:", args)

	# Load RNA-FM model
	model, alphabet = fm.pretrained.rna_fm_t12()
	batch_converter = alphabet.get_batch_converter()
	model.eval()  # disables dropout for deterministic results

	# load the dataset
	if args.reference_sequences is not None:
		reference = pd.read_csv(args.reference_sequences)

	# create column with corresponding mutation file path within reference dataframe
	reference['path'] = reference['fitness filename'].apply(lambda x: create_dms_file_path(x))

	# replace all T with U in sequences
	reference['RNA Construct Sequence'] = reference['Raw Construct Sequence'].str.replace('T', 'U')


	for index, row in reference.iterrows():
	    name = row['fitness filename']
	    wt_rna = row['RNA Construct Sequence'].upper()
	    csv_path = row['path']

	    # prepare data
	    data = [(name, wt_rna)]
	    batch_labels, batch_strs, batch_tokens = batch_converter(data)

	    # Extract embeddings (on CPU)
	    with torch.no_grad():
	        results = model(batch_tokens, repr_layers=[12])
	    token_embeddings = results["representations"][12]
	    token_logits = results["logits"]

	    # get logits
	    with torch.no_grad():
	        token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)

	    # read in mutations
	    mut_df = pd.read_csv(csv_path)
	    mut_df = mut_df[~mut_df['mutant'].isna()] # drop na
	    mut_df['mutant'] = mut_df['mutant'].str.replace('T', 'U') # RNA formatting

	    # create list of mutations
	    mut_list = mut_df['mutant'].to_list()

	    # score mutations
	    scores = []
	    for mut in mut_list:
	        scores.append(label_row(mut, wt_rna, token_probs, alphabet,1))

	    # append scores to df
	    mut_df['RNA_FM_scores'] = scores

	    # save df
	    score_path = f'{args.output_directory}{name}.csv'
	    mut_df.to_csv(score_path, index=False)
	    print('File saved at', score_path)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    sys.path.insert(1, args.model_location)
    import fm
    main(args)


