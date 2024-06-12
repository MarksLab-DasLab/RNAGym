import pandas as pd
from Bio import pairwise2
import os
import tqdm
from collections import defaultdict

def filter_sequences(sequences, public_sequences, threshold=0.2):
    """
    Filters sequences from sequences.csv to keep only "Private" sequences
    that are not 20% or more similar to any sequence in public_sequences.csv
    after alignment.

    Args:
        sequences: Pandas dataframe containing sequences information
        public_sequences: Pandas dataframe containing public sequences information
        threshold: Minimum identity level for filtering (default 0.2)

    Returns:
        Pandas dataframe containing filtered sequences
    """
    # Convert public sequences to a list for faster iteration
    public_seqs = public_sequences['sequence'].tolist()
    # Initialize a list to keep track of indices to keep
    to_keep = []
    # Iterate over each private sequence
    for idx, private_row in tqdm.tqdm(sequences.iterrows(), total=sequences.shape[0], desc="Filtering sequences"):
        seq1 = private_row["sequence"]
        keep = True  # Assume we keep the sequence unless proven otherwise
        for seq2 in public_seqs:
            # Perform global alignment between seq1 and seq2
            alignments = pairwise2.align.globalxx(seq1, seq2)
            # Check the best alignment score
            if alignments:
                # Get the best alignment (highest score)
                best_alignment = max(alignments, key=lambda x: x.score)
                identity = best_alignment.score / max(len(seq1), len(seq2))
                if identity >= threshold:
                    keep = False
                    break
        if keep:
            to_keep.append(idx)
    # Return the filtered DataFrame
    return sequences.loc[to_keep].copy()

def cluster_sequences(sequences, threshold=0.2):
  """
  Clusters sequences based on 20% identity threshold using BLAST

  Args:
      sequences: Pandas dataframe containing sequences information
      threshold: Minimum identity level for clustering (default 0.2)

  Returns:
      A list of lists, where each inner list represents a cluster of sequences
  """
  clusters = defaultdict(list)
  clusters_representatives = []
  for i, row_i in tqdm.tqdm(sequences.iterrows(),total=len(sequences)):
    seq1 = row_i["sequence"]
    already_clustered = False
    #for j, row_j in sequences.iloc[i + 1:].iterrows():
    for j, row_j in sequences.iterrows():
      print(j)
      seq2 = row_j["sequence"]
      # Perform pairwise alignment using BLAST
      alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -float("inf"), -float("inf"))
      score = alignments[0][2] / len(seq1)  # Identity as score / sequence length
      print(score)
      if score >= threshold:
        if clusters.get(seq1):
          clusters[seq1].append(seq2)
          already_clustered = True
        elif clusters.get(seq2):
          clusters[seq2].append(seq1)
          already_clustered = True
        else:
          clusters[seq1] = [seq1, seq2]
    if not already_clustered:
      clusters[seq1] = [seq1]
      clusters_representatives.append(seq1) 
  return clusters, clusters_representatives


if __name__ == "__main__":
    # Read the datasets
    data_folder = "../local_data"
    processed_folder = "../local_data/processed"
    test_data_name = "final_test_set.csv"
    filtered_data_name = "filtered_private_clusters_representatives_df"

    solution_df = pd.read_parquet(os.path.join(data_folder,'solution_ribonanza.parquet')) #len: 269,796,671
    ribonanzanet_df = pd.read_parquet(os.path.join(data_folder,'ribonanzanet.parquet'))
    test_sequences = pd.read_csv(os.path.join(data_folder,'test_sequences.csv')) #len: 1,343,823
    train_data = pd.read_csv(os.path.join(data_folder,'train_data.csv'))

    private_solution_df = solution_df[solution_df['Usage']=='Private'] #len: 14,902,527
    public_solution_df = solution_df[solution_df['Usage']=='Public'] #len: 6,188,100

    test_sequences['id'] = test_sequences.apply(lambda x: range(x['id_min'], x['id_max']+1), axis=1)
    test_sequences_exploded = test_sequences.explode('id') #length: 269,796,671
    solution_df = solution_df.merge(test_sequences_exploded[['id', 'sequence_id', 'sequence']], left_on='id', right_on='id', how='left') #len: 269,796,671

    # Filter test sequences based on 'Usage'
    private_seqs = solution_df[solution_df['Usage'] == 'Private'] #len: 14,902,527; 114,836 IDS
    public_seqs = solution_df[solution_df['Usage'] == 'Public'] #len: 6,188,100; 61,881 IDS
    set1 = set(private_seqs.sequence_id)
    set2 = set(public_seqs.sequence_id)
    assert len(set1 & set2)==0, "Sequence overlap between train and test" #0

    private_seqs.to_csv(os.path.join(processed_folder,test_data_name), index=False)
    private_seqs_unique_seqs = private_seqs.drop_duplicates('sequence') #len: 114,836
    
    # Filter parquet files for mutants in eval sequences
    solution_eval = solution_df[solution_df['sequence'].isin(private_seqs['sequence'])]
    ribonanzanet_eval = ribonanzanet_df[ribonanzanet_df['id'].isin(solution_eval['id'])]

    # Save filtered datasets
    solution_eval.to_csv(os.path.join(processed_folder,'solution_ribonanza_eval.csv'), index=False)
    ribonanzanet_eval.rename(columns={'reactivity_DMS_MaP': 'prediction_ribonanzanet'}).to_csv(os.path.join(processed_folder,'ribonanzanet_eval.csv'), index=False)

    # Clustering analysis
    private_clusters, private_clusters_representatives = cluster_sequences(private_seqs_unique_seqs, threshold=0.2)
    private_clusters_representatives_df = pd.DataFrame(private_clusters_representatives,columns=['sequence'])
    private_clusters_representatives_df.to_csv(os.path.join(processed_folder,"private_clusters_representatives_df.csv"), index=False)

    # Filtering analysis
    train_data_unique_seqs = train_data['sequence'].unique()
    train_data_unique_seqs = pd.DataFrame(train_data_unique_seqs,columns=['sequence']) #len: 806,573
    train_data_unique_seqs.to_csv(os.path.join(processed_folder,"train_data_unique_seqs.csv"), index=False)

    for threshold in range(2,10,2):
      threshold = threshold / 10
      print("Filtering at {}%".format(threshold*100))
      filtered_data_name_ext = filtered_data_name + "_" + str(threshold) + ".csv"
      filtered_private_seqs = filter_sequences(private_seqs_unique_seqs,train_data_unique_seqs,threshold)
      filtered_private_seqs.to_csv(os.path.join(processed_folder,filtered_data_name_ext), index=False)
