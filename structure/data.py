import os,sys
import pandas as pd

if __name__ == "__main__":
    # Read the datasets
    data_folder = sys.argv[1] # Location of raw & processed datasets (eg., "./local_data/structure_prediction")
    raw_data_folder = os.path.join(data_folder,'raw_data')
    processed_folder = os.path.join(data_folder,'test_data')
    model_predictions_folder = os.path.join(data_folder,'model_predictions')
    test_data_name = "final_test_set.csv"
    filtered_data_name = "filtered_private_clusters_representatives_df"

    solution_df = pd.read_parquet(os.path.join(raw_data_folder,'solution_ribonanza.parquet')) #len: 269,796,671 #Dataset with labels for all sequences / positions
    ribonanzanet_df = pd.read_parquet(os.path.join(raw_data_folder,'ribonanzanet.parquet')) #len: 269,796,671 #Predictions from Ribonanzanet
    test_sequences = pd.read_csv(os.path.join(raw_data_folder,'test_sequences.csv')) #len: 1,343,823 #Test dataset with actual RNA sequences
    train_data = pd.read_csv(os.path.join(raw_data_folder,'train_data.csv')) #len: 1,643,680 #Train dataset with actual RNA sequences

    test_sequences['id'] = test_sequences.apply(lambda x: range(x['id_min'], x['id_max']+1), axis=1)
    test_sequences_exploded = test_sequences.explode('id') #length: 269,796,671
    solution_df = solution_df.merge(test_sequences_exploded[['id', 'sequence_id', 'sequence']], left_on='id', right_on='id', how='left') #len: 269,796,671
    solution_df.sort_values(by=['sequence_id', 'id'], inplace=True)
    solution_df['position_id'] = solution_df.groupby('sequence_id').cumcount()
    solution_df.sort_values(by=['id'], inplace=True)
    
    # Filter test sequences based on 'Usage'
    #private_seqs_seq_id = solution_df[solution_df['Usage'] == 'Private'].sequence_id #len: 14,902,527
    private_seqs = solution_df[solution_df['Usage'] == 'Private'] #len: 14,902,527; 114,836 IDS
    public_seqs = solution_df[solution_df['Usage'] == 'Public'] #len: 6,188,100; 61,881 IDS
    set1 = set(private_seqs.sequence_id)
    set2 = set(public_seqs.sequence_id)
    assert len(set1 & set2)==0, "Sequence overlap between train and test" #0

    private_seqs.to_csv(os.path.join(processed_folder,test_data_name), index=False)
    private_seqs_unique_seqs = private_seqs.drop_duplicates('sequence') #len: 114,836
    
    # Filter parquet files for mutants in eval sequences
    solution_eval = solution_df[solution_df['sequence'].isin(private_seqs['sequence'])] #len: 23,740,972
    ribonanzanet_eval = ribonanzanet_df[ribonanzanet_df['id'].isin(solution_eval['id'])] #len: 23,740,972

    # Save filtered datasets
    solution_eval.to_csv(os.path.join(processed_folder,'solution_ribonanza_eval.csv'), index=False)
    ribonanzanet_eval.rename(columns={'reactivity_DMS_MaP': 'prediction_ribonanzanet'}).to_csv(os.path.join(model_predictions_folder,'predictions_ribonanzanet.csv'), index=False)
