import os
import pandas as pd
from pathlib import Path
import argparse
import logging

# Set up logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

# Function to identify the mutation column
def get_mutation_column(df):
    possible_columns = ['mutant', 'mutation', 'Mutation', 'mutations']
    for col in possible_columns:
        if col in df.columns:
            return col
    raise ValueError("Couldn't find a mutation column in the dataframe")

# Function to standardize mutation strings (replace 'T' with 'U')
def standardize_mutation(mutation):
    return str(mutation).replace('T', 'U')

# Function to combine CSV data with model predictions
def combine_csv_data(processed_folder, model_predictions_folder, output_folder, model_list, score_cols_dict):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for csv_file in os.listdir(processed_folder):
        if csv_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(processed_folder, csv_file))
            mutation_col = get_mutation_column(df)
            df = df.dropna(subset=[mutation_col])
            df[mutation_col] = df[mutation_col].apply(standardize_mutation)

            save = True
            for model_name in model_list:
                model_path = os.path.join(model_predictions_folder, model_name, csv_file)
                if os.path.exists(model_path):
                    model_df = pd.read_csv(model_path)
                    model_mutation_col = get_mutation_column(model_df)
                    score_col = score_cols_dict[model_name] #model_df.columns[-1]  # Assume the last column is the score
                    model_df = model_df[[model_mutation_col, score_col]]
                    model_df.columns = [mutation_col, f'{model_name}_score']
                    model_df[mutation_col] = model_df[mutation_col].apply(standardize_mutation)
                    model_df = model_df.dropna(subset=[mutation_col])
                    model_df = model_df.drop_duplicates(subset=[mutation_col], keep='first')

                    original_row_count = len(df)
                    df = df.merge(model_df, on=mutation_col, how='inner')
                    merged_row_count = len(df)
                    if original_row_count != merged_row_count:
                        #raise ValueError(f"Row count mismatch after merging {model_name}: Expected {original_row_count}, but got {merged_row_count}")
                        print(f"Row count mismatch in {csv_file} after merging {model_name}: Expected {original_row_count}, but got {merged_row_count}")
                        save = False
                else:
                    logger.warning(f"Model file {csv_file} not found in {model_name}")
                    save = False

            if len(df) > 0 and save:
                output_file = os.path.join(output_folder, csv_file)
                df.to_csv(output_file, index=False)
                logger.info(f"Saved combined data to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine processed CSV data with model predictions.")
    parser.add_argument("--processed_folder", required=True, help="Path to the folder containing processed CSV files.")
    parser.add_argument("--model_predictions_folder", required=True, help="Path to the folder containing model prediction subfolders.")
    parser.add_argument("--output_folder", required=True, help="Path to the folder where combined CSV files will be saved.")
    parser.add_argument('--assays_with_MSAs_only', action='store_true', help='Focus on assays with MSAs only (eg., PSSM, EVmutation)')
    
    args = parser.parse_args()

    if args.assays_with_MSAs_only:
        model_list = ['PSSM', 'EVmutation']
    else:
        model_list = ['Evo','GenSLM','Nucleotide_Transformer','RNA_FM','RiNALMo']
    
    score_cols_dict = {
        'Evo': 'scores',
        'GenSLM': 'logit_scores',
        'Nucleotide_Transformer': 'logit_scores',
        'RNA_FM': 'RNA_FM_scores',
        'RiNALMo': 'logit_scores',
        'PSSM': 'prediction_independent',
        'EVmutation': 'prediction_epistatic'
    }

    combine_csv_data(args.processed_folder, args.model_predictions_folder, args.output_folder, model_list, score_cols_dict)

if __name__ == "__main__":
    main()