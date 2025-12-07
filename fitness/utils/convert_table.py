import csv
import argparse
import re
from pathlib import Path
from Bio import Data
from Bio.Data import CodonTable


def parse_fasta(fasta_file):
    """Parse a FASTA file and return the sequence."""
    sequence = ""
    with open(fasta_file, "r") as f:
        for line in f:
            if not line.startswith(">"):  # Skip header lines
                sequence += line.strip()
    return sequence.upper()


def get_codon_table(table_id=1):
    """Get the codon table based on NCBI table ID.

    Args:
        table_id (int): NCBI genetic code table ID
                        1 = Standard (default)
                        2 = Vertebrate Mitochondrial
                        3 = Yeast Mitochondrial
                        4 = Mold, Protozoan, Coelenterate Mitochondrial and Mycoplasma/Spiroplasma
                        5 = Invertebrate Mitochondrial
                        6 = Ciliate Nuclear and Dasycladacean
                        9 = Echinoderm Mitochondrial and Flatworm Mitochondrial
                        10 = Euplotid Nuclear
                        11 = Bacterial, Archaeal and Plant Plastid
                        12 = Alternative Yeast Nuclear
                        13 = Ascidian Mitochondrial
                        14 = Alternative Flatworm Mitochondrial
                        16 = Chlorophycean Mitochondrial
                        21 = Trematode Mitochondrial
                        22 = Scenedesmus obliquus Mitochondrial
                        23 = Thraustochytrium Mitochondrial
                        24 = Rhabdopleuridae Mitochondrial
                        25 = Candidate Division SR1 and Gracilibacteria
                        26 = Pachysolen tannophilus Nuclear
                        27 = Karyorelict Nuclear
                        28 = Condylostoma Nuclear
                        29 = Mesodinium Nuclear
                        30 = Peritrich Nuclear
                        31 = Blastocrithidia Nuclear
                        33 = Cephalodiscidae Mitochondrial
    """
    biopython_table = CodonTable.unambiguous_dna_by_id[table_id]

    # Create our own codon table dictionary from the Biopython table
    codon_table = {}
    for codon, aa in biopython_table.forward_table.items():
        codon_table[codon] = aa

    # Add stop codons
    for stop_codon in biopython_table.stop_codons:
        codon_table[stop_codon] = "*"

    return codon_table


def get_reverse_codon_table(codon_table):
    """Create reverse codon table (amino acid to possible codons)."""
    reverse_codon_table = {}
    for codon, aa in codon_table.items():
        if aa not in reverse_codon_table:
            reverse_codon_table[aa] = []
        reverse_codon_table[aa].append(codon)
    return reverse_codon_table


def list_available_codon_tables():
    """List all available codon tables with names and IDs."""
    print("Available codon tables:")
    print("ID | Name")
    print("---|-------------------")
    for table_id in sorted(CodonTable.unambiguous_dna_by_id.keys()):
        table = CodonTable.unambiguous_dna_by_id[table_id]
        print(f"{table_id:2d} | {table.names[0]}")


def translate_dna(dna_sequence, codon_table):
    """Translate DNA sequence to protein using specified codon table."""
    protein = ""
    for i in range(0, len(dna_sequence), 3):
        if i + 3 <= len(dna_sequence):
            codon = dna_sequence[i : i + 3]
            protein += codon_table.get(codon, "X")
    return protein


def parse_mutation(mutation_code):
    """Parse mutation code like 'H24S' into original AA, position, and new AA."""
    mutation_code = mutation_code.strip('" ')  # Remove quotes and spaces
    if not re.match(r"^[A-Z]\d+[A-Z]$", mutation_code):
        raise ValueError(f"Invalid mutation format: {mutation_code}")

    orig_aa = mutation_code[0]
    # Parse the position but keep it as 1-based for validation against the protein sequence
    pos_1based = int(mutation_code[1:-1])
    new_aa = mutation_code[-1]

    return orig_aa, pos_1based, new_aa


def apply_mutation_and_get_nucleotide_change(
    original_dna, position_1based, new_aa, original_protein, reverse_codon_table
):
    """Apply a protein mutation to DNA and return both the nucleotide changes and the new DNA."""
    # Convert to 0-based for internal use
    pos_0based = position_1based - 1

    # Check position validity
    if pos_0based < 0 or pos_0based >= len(original_protein):
        raise ValueError(
            f"Position {position_1based} is out of range for protein length {len(original_protein)}"
        )

    # Find the corresponding DNA position (3 nucleotides per amino acid)
    dna_pos = pos_0based * 3

    # Get the original codon at this position
    original_codon = original_dna[dna_pos : dna_pos + 3]

    # Check if the new_aa is in the reverse_codon_table
    if new_aa not in reverse_codon_table:
        raise ValueError(f"Amino acid '{new_aa}' not found in the codon table")

    # For consistency with the DMS data, use the same codon pattern when possible
    # We'll try to change the minimum number of nucleotides
    best_codon = None
    min_changes = 3

    for new_codon in reverse_codon_table[new_aa]:
        changes = sum(1 for a, b in zip(original_codon, new_codon) if a != b)
        if changes <= min_changes:
            min_changes = changes
            best_codon = new_codon

    if best_codon is None:
        # This shouldn't happen now with the <= operator, but let's keep it as a safety check
        print(
            f"Warning: No suitable codon found for {new_aa} at position {position_1based}"
        )
        best_codon = reverse_codon_table[new_aa][0]  # Use first available codon

    # Identify specific nucleotide changes
    nt_changes = []
    for i in range(3):
        if original_codon[i] != best_codon[i]:
            nt_change = f"{original_codon[i]}{dna_pos+i+1}{best_codon[i]}"
            nt_changes.append(nt_change)

    # Create the new DNA sequence
    new_dna = original_dna[:dna_pos] + best_codon + original_dna[dna_pos + 3 :]

    return nt_changes, new_dna


def process_mutations_file(
    dms_file, output_file, original_dna, original_protein, reverse_codon_table
):
    """Process the mutations file and convert protein mutations to DNA sequences."""
    # Create output file
    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["mutant", "DMS_score", "sequence"])

        # Read the input file
        with open(dms_file, "r") as infile:
            reader = csv.DictReader(infile)

            # Track progress
            total_rows = sum(1 for _ in open(dms_file)) - 1  # Subtract header
            processed = 0
            errors = 0

            for row in reader:
                try:
                    protein_mutant = row.get("mutant", "").strip('" ')
                    dms_score = row.get("DMS_score", "")

                    # If no mutation, skip
                    if not protein_mutant:
                        continue

                    # Split mutations - now handling both comma and colon separators
                    protein_mutations = []
                    if "," in protein_mutant:
                        protein_mutations = protein_mutant.split(",")
                    elif ":" in protein_mutant:
                        protein_mutations = protein_mutant.split(":")
                    else:
                        # Single mutation
                        protein_mutations = [protein_mutant]

                    current_dna = original_dna
                    all_nt_changes = []

                    for mut in protein_mutations:
                        mut = mut.strip('" ')  # Remove quotes and spaces
                        if not mut:  # Skip empty mutations
                            continue

                        orig_aa, pos_1based, new_aa = parse_mutation(mut)
                        pos_0based = pos_1based - 1

                        # Verify the original amino acid matches
                        if (
                            0 <= pos_0based < len(original_protein)
                            and original_protein[pos_0based] != orig_aa
                        ):
                            print(
                                f"Mismatch at position {pos_1based}: expected {orig_aa}, found {original_protein[pos_0based]}"
                            )
                            raise ValueError(
                                f"Mismatch at position {pos_1based}: expected {orig_aa}, found {original_protein[pos_0based]}"
                            )

                        # Apply the mutation to DNA and get nucleotide changes
                        nt_changes, current_dna = (
                            apply_mutation_and_get_nucleotide_change(
                                current_dna,
                                pos_1based,
                                new_aa,
                                original_protein,
                                reverse_codon_table,
                            )
                        )
                        all_nt_changes.extend(nt_changes)

                    # Convert nucleotide changes to comma-separated string
                    nt_mutant = ",".join(all_nt_changes)

                    # Write the output row with nucleotide mutations
                    writer.writerow([nt_mutant, dms_score, current_dna])

                except Exception as e:
                    print(
                        f"Error processing mutation {row.get('mutant', 'unknown')}: {e}"
                    )
                    # Write the row with ERROR in sequence field
                    writer.writerow([protein_mutant, dms_score, "ERROR"])
                    errors += 1

                processed += 1
                if processed % 100 == 0:
                    print(
                        f"Processed {processed}/{total_rows} mutations ({processed/total_rows*100:.1f}%)"
                    )

            print(f"Completed processing {processed} mutations with {errors} errors.")


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Convert protein mutations to nucleotide sequences."
    )
    parser.add_argument("fasta_file", help="Input FASTA file with nucleotide sequence")
    parser.add_argument("dms_file", help="Input DMS file with mutations")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file (default: input_with_nucleotides.csv)",
    )
    parser.add_argument(
        "--table",
        "-t",
        type=int,
        default=1,
        help="NCBI genetic code table ID (default: 1, Standard)",
    )
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List all available codon tables and exit",
    )
    args = parser.parse_args()

    # List tables if requested
    if args.list_tables:
        list_available_codon_tables()
        return

    # Determine output filename if not specified
    if args.output is None:
        output_file = Path(args.dms_file).stem + "_with_sequences.csv"
    else:
        output_file = args.output

    # Get the appropriate codon table
    codon_table = get_codon_table(args.table)
    reverse_codon_table = get_reverse_codon_table(codon_table)

    # Print the selected table
    table_name = CodonTable.unambiguous_dna_by_id[args.table].names[0]
    print(f"Using codon table {args.table}: {table_name}")

    print(f"Reading nucleotide sequence from {args.fasta_file}")
    original_dna = parse_fasta(args.fasta_file)
    print(f"Nucleotide sequence length: {len(original_dna)} bp")

    # Translate to protein
    original_protein = translate_dna(original_dna, codon_table)
    print(original_protein)
    print(f"Translated protein length: {len(original_protein)} amino acids")
    print(f"First 10 amino acids: {original_protein[:10]}")
    print(f"Last 10 amino acids: {original_protein[-10:]}")

    # Process the mutations file
    print(f"\nProcessing mutations from {args.dms_file}")
    process_mutations_file(
        args.dms_file, output_file, original_dna, original_protein, reverse_codon_table
    )
    print(original_dna)
    print(f"Processing complete. Output saved to {output_file}")


if __name__ == "__main__":
    main()
