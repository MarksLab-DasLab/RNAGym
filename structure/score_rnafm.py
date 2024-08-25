import pandas as pd
import numpy as np
import os,sys
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error
import tqdm
import torch
import matplotlib.pyplot as plt
import fm
import torch.nn.functional as F

def unpaired_probabilities(prob_matrix):
    unpaired_probs = torch.prod(1 - prob_matrix, dim=1)
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

model_type='RNA_FM'
data_folder = 'Path to model predictions' #Replace w/ your folder path
test_data_folder = os.path.join(data_folder,'test_data')
final_output_repo = os.path.join(data_folder,'model_predictions')
final_test_data_name = "final_test_set.csv"
performance_summary_file = 'performance_SS_pred.csv'
final_test_set_all_mutants = pd.read_csv(os.path.join(test_data_folder,final_test_data_name), low_memory=False)

# Load RNA-FM model
model, alphabet = fm.downstream.build_rnafm_resnet(type="ss")
batch_converter = alphabet.get_batch_converter()
model.eval()  # Disables dropout for deterministic results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is on device: {device}")


def score_rna_fm(batch_data):
    batch_data = [ tuple(data.values()) for data in batch_data]
    batch_labels, _, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.cuda()  # Move data to GPU

    input = {
        "description": batch_labels,
        "token": batch_tokens
    }

    with torch.no_grad():
        results = model(input)

    logits = results["r-ss"]
    probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities
    return probabilities

# Prepare the data and process in batches
final_test_set_all_seqs = final_test_set_all_mutants[["sequence_id", "sequence"]].drop_duplicates()
batch_size = 32  # Adjust batch size as needed

all_predictions = []
for i in tqdm.tqdm(range(0, len(final_test_set_all_seqs), batch_size)):
    batch_data = final_test_set_all_seqs.iloc[i:i + batch_size].to_dict('records')
    batch_predictions = score_rna_fm(batch_data)
    
    for j, row in enumerate(batch_data):
        pred = unpaired_probabilities(batch_predictions[j])
        pred_df = {
            'sequence': [row['sequence']] * len(pred),
            'sequence_id': [row['sequence_id']] * len(pred),
            'position_id': range(len(pred)),
            'prediction_{}'.format(model_type): pred.cpu().numpy()  # Move data back to CPU for DataFrame
        }
        all_predictions.append(pd.DataFrame(pred_df))

all_predictions = pd.concat(all_predictions)
all_predictions['sequence_id'] = all_predictions['sequence_id'].apply(lambda x: str(x))
all_predictions['position_id'] = all_predictions['position_id'].apply(lambda x: int(x))
final_test_set_all_mutants['sequence_id'] = final_test_set_all_mutants['sequence_id'].apply(lambda x: str(x))
final_test_set_all_mutants['position_id'] = final_test_set_all_mutants['position_id'].apply(lambda x: int(x))

all_predictions.to_csv(final_output_repo+os.sep+'predictions_'+model_type+'_all_predictions.csv',index=False)
# Merge and save 
final_test_set_all_mutants_pred = pd.merge(final_test_set_all_mutants, all_predictions, on=['sequence_id',"position_id"], how="left")
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

