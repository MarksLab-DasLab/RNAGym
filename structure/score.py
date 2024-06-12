import pandas as pd
import numpy as np
import os,sys
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error
import tqdm
os.environ['ARNIEFILE'] = './structure/arniefile.txt'
from arnie.bpps import bpps

def unpaired_probabilities(prob_matrix):
    _, num_cols = prob_matrix.shape
    unpaired_probs = np.array([np.prod(1 - prob_matrix[i]) for i in range(num_cols)])
    return unpaired_probs

def predict_structures(seq, models=['eternafold', 'contrafold_2', 'vienna2', 'rnastructure']):
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
    model_type = sys.argv[1]
    print("Model type: {}".format(model_type))
    final_output_repo = "../local_data"
    final_test_data_name = "final_test_set.csv"
    performance_summary_file = 'performance_SS_pred.csv'

    final_test_set_all_mutants = pd.read_csv(os.path.join(final_output_repo,final_test_data_name), low_memory=False)
    final_test_set_all_mutants.sort_values(by=['sequence_id', 'id'], inplace=True)
    final_test_set_all_mutants['position_id'] = final_test_set_all_mutants.groupby('sequence_id').cumcount()

    final_test_set_all_seqs = final_test_set_all_mutants[["sequence_id","sequence"]].drop_duplicates()
    print(len(final_test_set_all_seqs)) #114,836

    # Predict with selected model
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

    # Merge and preprocess
    final_test_set_all_mutants_pred = pd.merge(final_test_set_all_mutants,all_models_predictions, on=['sequence_id',"position_id"], how="left")
    final_test_set_all_mutants_pred.to_csv(final_output_repo+os.sep+'predictions_'+model_type,index=False)
    final_test_set_all_mutants_pred = final_test_set_all_mutants_pred[final_test_set_all_mutants_pred['prediction_{}'.format(model_type)].notna()]
    final_test_set_all_mutants_pred = final_test_set_all_mutants_pred[final_test_set_all_mutants_pred["Usage"] != "Ignored"]
    final_test_set_all_mutants_pred['reactivity_DMS_MaP_bin'] = final_test_set_all_mutants_pred['reactivity_DMS_MaP'] > final_test_set_all_mutants_pred['reactivity_DMS_MaP'].median()
    final_test_set_all_mutants_pred['prediction_{}_bin'.format(model_type)] = final_test_set_all_mutants_pred['prediction_{}'.format(model_type)] > final_test_set_all_mutants_pred['prediction_{}'.format(model_type)].median()

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