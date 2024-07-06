import pandas as pd
import numpy as np
import os,sys
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error
import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['ARNIEFILE'] = os.path.join(script_dir, 'arniefile.txt')
from arnie.bpps import bpps

def unpaired_probabilities(prob_matrix):
    _, num_cols = prob_matrix.shape
    unpaired_probs = np.array([np.prod(1 - prob_matrix[i]) for i in range(num_cols)])
    return unpaired_probs

def compute_bins(df, model_type):
    # Compute median for reactivity_DMS_MaP and create binary column
    median_reactivity = df['reactivity_DMS_MaP'].median()
    df['reactivity_DMS_MaP_bin'] = df['reactivity_DMS_MaP'] > median_reactivity
    # Compute median for prediction and create binary column
    prediction_col = f'prediction_{model_type}'
    median_prediction = df[prediction_col].median()
    df[f'{prediction_col}_bin'] = df[prediction_col] > median_prediction
    return df

def predict_structures(seq, models=['eternafold', 'contrafold_2', 'vienna_2', 'rnastructure']):
    """
    Predict RNA secondary structures using specified models and track progress.

    Args:
        seq (str): The RNA sequence for which to predict structures.
        models (list of str): List of model names to use for predictions.

    Returns:
        dict: A dictionary with model names as keys and prediction results as values.
    """
    predictions = {}
    for model in models:
        predictions[model] = bpps(seq, package=model)
    return predictions

if __name__ == "__main__":
    model_type = sys.argv[1] #"eternafold"
    print("Model type: {}".format(model_type))
    data_folder = sys.argv[2] #"./local_data/structure_prediction"
    if model_type == "rnastructure":
        os.environ['DATAPATH'] = os.path.join(data_folder,'models/RNAstructure/data_tables')
    test_data_folder = os.path.join(data_folder,'test_data')
    final_output_repo = os.path.join(data_folder,'model_predictions')
    final_test_data_name = "final_test_set.csv"
    performance_summary_file = 'performance_SS_pred.csv'
    final_test_set_all_mutants = pd.read_csv(os.path.join(test_data_folder,final_test_data_name), low_memory=False)
    
    # Predict with selected model
    if os.path.exists(final_output_repo+os.sep+'predictions_'+model_type+'.csv'):
        final_test_set_all_mutants_pred = pd.read_csv(final_output_repo+os.sep+'predictions_'+model_type+'.csv')
        if model_type=="ribonanzanet":
            final_test_set_all_mutants_pred = final_test_set_all_mutants_pred[['id','prediction_ribonanzanet']]
            final_test_set_all_mutants_pred = pd.merge(final_test_set_all_mutants,final_test_set_all_mutants_pred, on='id', how="left") #len: 14,902,527
    else:
        final_test_set_all_seqs = final_test_set_all_mutants[["sequence_id","sequence"]].drop_duplicates()
        print(len(final_test_set_all_seqs)) #len: 114,836
        all_models_predictions = []
        for row_index, row in tqdm.tqdm(final_test_set_all_seqs.iterrows(), total=len(final_test_set_all_seqs)):
            pred = predict_structures(row['sequence'], models=[model_type])[model_type]
            pred = unpaired_probabilities(pred)
            pred_df = {
                'sequence': [row['sequence']] * len(pred),
                'sequence_id': [row['sequence_id']] * len(pred),
                'position_id': range(len(pred)),
                'prediction_{}'.format(model_type) : pred
            }
            pred_df = pd.DataFrame(pred_df)
            all_models_predictions.append(pred_df)
        all_models_predictions = pd.concat(all_models_predictions)
        # Merge and save
        final_test_set_all_mutants_pred = pd.merge(final_test_set_all_mutants,all_models_predictions, on=['sequence_id',"position_id"], how="left")
        final_test_set_all_mutants_pred.to_csv(final_output_repo+os.sep+'predictions_'+model_type+'.csv',index=False)
    
    # Process for scoring
    final_test_set_all_mutants_pred = final_test_set_all_mutants_pred[final_test_set_all_mutants_pred['prediction_{}'.format(model_type)].notna()]
    final_test_set_all_mutants_pred = final_test_set_all_mutants_pred[final_test_set_all_mutants_pred["Usage"] != "Ignored"]
    final_test_set_all_mutants_pred['reactivity_DMS_MaP'] = final_test_set_all_mutants_pred['reactivity_DMS_MaP'].apply(lambda x: max(min(x,1.0),0.0))
    final_test_set_all_mutants_pred['prediction_{}'.format(model_type)] = final_test_set_all_mutants_pred['prediction_{}'.format(model_type)].apply(lambda x: max(min(x,1.0),0.0))
    final_test_set_all_mutants_pred = final_test_set_all_mutants_pred.groupby('sequence_id', group_keys=False).apply(compute_bins, model_type=model_type)

    # Compute performance
    f1_score_model = f1_score(
        final_test_set_all_mutants_pred['reactivity_DMS_MaP_bin'], 
        final_test_set_all_mutants_pred['prediction_{}_bin'.format(model_type)], 
        average='macro'
    )
    auc_score = roc_auc_score(
        final_test_set_all_mutants_pred['reactivity_DMS_MaP_bin'],
        final_test_set_all_mutants_pred['prediction_{}'.format(model_type)]
    )
    mcc_score = matthews_corrcoef(
        final_test_set_all_mutants_pred['reactivity_DMS_MaP_bin'],
        final_test_set_all_mutants_pred['prediction_{}_bin'.format(model_type)]
    )
    mae_score = mean_absolute_error(
        final_test_set_all_mutants_pred['reactivity_DMS_MaP'],
        final_test_set_all_mutants_pred['prediction_{}'.format(model_type)]
    )
    header = not os.path.exists(performance_summary_file)
    metrics_df = pd.DataFrame({
        'Model Type': [model_type],
        'AUC': [auc_score],
        'MCC': [mcc_score],
        'MAE': [mae_score],
        'F1-Score': [f1_score_model]
    })
    print(metrics_df)
    metrics_df.to_csv(performance_summary_file, mode='a', index=False, header=header)