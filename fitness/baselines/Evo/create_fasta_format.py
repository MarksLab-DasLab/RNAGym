import argparse
import pandas as pd
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def get_sequences(wt_sequence, df):
    
    # Function to apply a single mutation
    def apply_mutation(sequence, mutation, convert_to_upper=True):
        if convert_to_upper:
            sequence = sequence.upper()
        sequence = sequence.replace('U', 'T')
        possible_bases = ['A', 'T', 'C', 'G', 'N', '']
        mutation = mutation.replace(' ', '')
        pos = int(mutation[1:-1]) - 1  # Get the position, 1-based to 0-based index
        new_base = mutation[-1]  # Get the new base
        old_base = mutation[0]
        old_base = 'T' if old_base == 'U' else old_base
        new_base = 'T' if new_base == 'U' else new_base
        
        assert old_base in possible_bases, mutation
        assert new_base in possible_bases, mutation

        # if old_base == 'N':
        #     # This is going to be an insertion
        #     mutated_sequence = sequence[:pos] + new_base + sequence[pos+1:]
        if new_base == '':
            # This is going to be a deletion
            mutated_sequence = sequence[:pos] + sequence[pos+1:]
        else:
            # This is a substitution
            if old_base == sequence[pos]:
                mutated_sequence = sequence[:pos] + new_base + sequence[pos+1:]
            else:
                print(f"Skipping mutation: {mutation}. Cannot apply mutation to sequence: {sequence}")
                return None
        return mutated_sequence


    # Function to apply multiple mutations
    def apply_mutations(sequence, mutations):
        # print(mutations, type(mutations))
        for mutation in mutations.split(','):
            sequence = apply_mutation(sequence, mutation)
            if sequence is None:
                continue
        return sequence.replace('N', '') # remove inser/delet fill in tokens

    # # Determine which column to use for mutations
    # mutation_column = 'mutant' if 'mutant' in df.columns else 'mutation' if 'mutation' in df.columns else 'mutations' if 'mutations' in df.columns else None

    # List of potential column names
    potential_columns = ['mutant', 'mutation', 'mutations']

    # Convert DataFrame column names to lowercase
    df.columns = df.columns.str.lower()

    # Determine which column to use for mutations
    mutation_column = next((col for col in potential_columns if col in df.columns), None)

    if mutation_column:
        # Remove rows with NaN in the mutation column
        df = df.dropna(subset=[mutation_column])
        
        # Ensure the mutation column is of type string
        df[mutation_column] = df[mutation_column].astype(str)

        # Apply the mutations to create a new column with mutated sequences
        df['mutated_sequence'] = df[mutation_column].apply(lambda x: apply_mutations(wt_sequence, x))
    else:
        raise ValueError("No 'mutant' or 'mutation' column found in the DataFrame")
    
    return df

def create_fasta_from_csv(ref_file, row):
    # Get the csv_filename and fitness_filename from the specified row in the reference file
    # print(f"Reading reference file: {ref_file}")
    ref_df = pd.read_csv(ref_file, encoding='latin-1')
    # print("Reference file read successfully with columns:", ref_df.columns)
    fitness_filename = ref_df.loc[row, 'fitness filename']
    wt_sequence = ref_df.loc[row, 'Raw Construct Sequence (DNA)']

    csv_filename = os.path.join(os.path.dirname(ref_file), "processed", fitness_filename + '.csv')

    # Read the CSV file
    # print(f"Reading CSV file: {csv_filename}")
    df = pd.read_csv(csv_filename, encoding='utf-8')
    # print(df.head())
    df = get_sequences(wt_sequence, df)

    # Create SeqRecord objects for each mutant
    records = [SeqRecord(Seq(mutant), id=f"{fitness_filename}_{i}", description="") for i, mutant in enumerate(df['mutated_sequence'], 1)]

    # Create the parent directory if it doesn't exist
    fasta_dir = os.path.join(os.path.dirname(ref_file), 'fasta_format')
    os.makedirs(fasta_dir, exist_ok=True)

    # Write the SeqRecord objects to a FASTA file
    fasta_file = os.path.join(fasta_dir, f"{fitness_filename}.fasta")
    SeqIO.write(records, fasta_file, "fasta")
    # print(f"FASTA file written to: {fasta_file}")
    return fitness_filename

def main():
    parser = argparse.ArgumentParser(description='Create a FASTA file from a CSV of mutants.')
    parser.add_argument('ref_file', help='The reference file.')
    parser.add_argument('row', type=int, help='The row to select the csv_filename and fitness_filename from.')
    args = parser.parse_args()

    fitness_filename = create_fasta_from_csv(args.ref_file, args.row)
    print(fitness_filename)
    return fitness_filename

if __name__ == "__main__":
    main()
