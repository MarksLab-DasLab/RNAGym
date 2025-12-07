import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from paddlenlp.transformers import ErnieForMaskedLM
import paddle
from src.rna_ernie import BatchConverter


# Function Definitions
def construct_file_path(directory, filename, extension=".csv"):
    """Constructs a full file path given a directory, filename, and extension."""
    return f"{directory}{filename}{extension}"


def calculate_mutation_score(mutation, sequence, token_probs, alphabet, offset):
    """Calculates the mutation score for a given mutation."""
    score = 0
    for mut in mutation.split(","):
        mut = mut.strip()
        wt, idx, mt = mut[0], int(mut[1:-1]) - offset, mut[-1]
        assert sequence[idx] == wt, "Mismatch between sequence and wildtype."

        wt_encoded, mt_encoded = alphabet[wt], alphabet[mt]
        score += (
            token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        ).item()
    return score


def main(args):
    # Load Model
    language_model = ErnieForMaskedLM.from_pretrained(args.model_checkpoint)
    language_model.eval()

    # Initialize BatchConverter
    batch_converter = BatchConverter(
        k_mer=1, vocab_path=args.vocab_path, batch_size=256, max_seq_len=512
    )

    # Load Reference Data
    reference_data = pd.read_csv(args.reference_sequences)
    reference_data["path"] = reference_data["DMS_ID"].apply(
        lambda x: construct_file_path(args.dms_directory, x)
    )
    reference_data["RAW_CONSTRUCT_SEQ"] = reference_data[
        "RAW_CONSTRUCT_SEQ"
    ].str.replace("U", "T")

    alphabet = batch_converter.tokenizer.vocab.token_to_idx

    # Processing Loop
    for idx, row in reference_data.iterrows():
        experiment_id = row["DMS_ID"]
        wildtype_sequence = row["RAW_CONSTRUCT_SEQ"].upper()
        mutation_file_path = row["path"]

        # Prepare Data for Model
        data_batch = [(experiment_id, wildtype_sequence)]

        for _, _, input_ids in batch_converter(data_batch):
            with paddle.no_grad():
                logits = language_model(input_ids).detach()
                logits_tensor = torch.tensor(
                    logits.numpy()
                )  # Convert Paddle tensor to PyTorch
                probabilities = F.softmax(logits_tensor, dim=2)

            # Load Mutations and Format
            mutation_data = pd.read_csv(mutation_file_path).dropna(subset=["mutant"])
            mutation_data["mutant"] = mutation_data["mutant"].str.replace("U", "T")
            mutation_list = mutation_data["mutant"].tolist()

            # Score Mutations
            mutation_scores = [
                calculate_mutation_score(
                    mut, wildtype_sequence, probabilities, alphabet, offset=1
                )
                for mut in mutation_list
            ]

            # Append Scores and Save
            mutation_data["Mutation_Scores"] = mutation_scores
            output_path = construct_file_path(args.output_directory, experiment_id)
            mutation_data.to_csv(output_path, index=False)
            print(f"Scores saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mutation scores using an RNA language model."
    )
    parser.add_argument(
        "--reference_sequences",
        type=str,
        required=True,
        help="Path to the reference sequences CSV file.",
    )
    parser.add_argument(
        "--dms_directory",
        type=str,
        required=True,
        help="Directory containing DMS files.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--vocab_path", type=str, required=True, help="Path to the vocabulary file."
    )

    args = parser.parse_args()
    main(args)
