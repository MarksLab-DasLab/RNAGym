import csv
import argparse
import re
from pathlib import Path

# Standard codon table (for E. coli - bacterial genetic code)
codon_table = {
    # Phenylalanine (F)
    'TTT': 'F', 'TTC': 'F',
    # Leucine (L)
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    # Isoleucine (I)
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    # Methionine (M) - Start
    'ATG': 'M',
    # Valine (V)
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    # Serine (S)
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    # Proline (P)
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Threonine (T)
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Alanine (A)
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Tyrosine (Y)
    'TAT': 'Y', 'TAC': 'Y',
    # Histidine (H)
    'CAT': 'H', 'CAC': 'H',
    # Glutamine (Q)
    'CAA': 'Q', 'CAG': 'Q',
    # Asparagine (N)
    'AAT': 'N', 'AAC': 'N',
    # Lysine (K)
    'AAA': 'K', 'AAG': 'K',
    # Aspartic Acid (D)
    'GAT': 'D', 'GAC': 'D',
    # Glutamic Acid (E)
    'GAA': 'E', 'GAG': 'E',
    # Cysteine (C)
    'TGT': 'C', 'TGC': 'C',
    # Tryptophan (W)
    'TGG': 'W',
    # Arginine (R)
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Glycine (G)
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Stop codons
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}

# Create reverse codon table (amino acid to possible codons)
reverse_codon_table = {}
for codon, aa in codon_table.items():
    if aa not in reverse_codon_table:
        reverse_codon_table[aa] = []
    reverse_codon_table[aa].append(codon)

def parse_fasta(fasta_file):
    """Parse a FASTA file and return the sequence."""
    sequence = ""
    with open(fasta_file, 'r') as f:
        for line in f:
            if not line.startswith('>'):  # Skip header lines
                sequence += line.strip()
    return sequence.upper()

def translate_dna(dna_sequence):
    """Translate DNA sequence to protein."""
    protein = ''
    for i in range(0, len(dna_sequence), 3):
        if i + 3 <= len(dna_sequence):
            codon = dna_sequence[i:i+3]
            protein += codon_table.get(codon, 'X')
    return protein

def parse_mutation(mutation_code):
    """Parse mutation code like 'H24S' into original AA, position, and new AA."""
    mutation_code = mutation_code.strip('" ')  # Remove quotes and spaces
    if not re.match(r'^[A-Z]\d+[A-Z]$', mutation_code):
        raise ValueError(f"Invalid mutation format: {mutation_code}")
    
    orig_aa = mutation_code[0]
    # Parse the position but keep it as 1-based for validation against the protein sequence
    pos_1based = int(mutation_code[1:-1])
    new_aa = mutation_code[-1]
    
    return orig_aa, pos_1based, new_aa

def apply_mutation_and_get_nucleotide_change(original_dna, position_1based, new_aa, original_protein):
    """Apply a protein mutation to DNA and return both the nucleotide changes and the new DNA."""
    # Convert to 0-based for internal use
    pos_0based = position_1based - 1
    
    # Check position validity
    if pos_0based < 0 or pos_0based >= len(original_protein):
        raise ValueError(f"Position {position_1based} is out of range for protein length {len(original_protein)}")
    
    # Find the corresponding DNA position (3 nucleotides per amino acid)
    dna_pos = pos_0based * 3
    
    # Get the original codon at this position
    original_codon = original_dna[dna_pos:dna_pos+3]
    
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
        print(f"Warning: No suitable codon found for {new_aa} at position {position_1based}")
        best_codon = reverse_codon_table[new_aa][0]  # Use first available codon
    
    # Identify specific nucleotide changes
    nt_changes = []
    for i in range(3):
        if original_codon[i] != best_codon[i]:
            nt_change = f"{original_codon[i]}{dna_pos+i+1}{best_codon[i]}"
            nt_changes.append(nt_change)
    
    # Create the new DNA sequence
    new_dna = original_dna[:dna_pos] + best_codon + original_dna[dna_pos+3:]
    
    return nt_changes, new_dna

def process_mutations_file(dms_file, output_file, original_dna, original_protein):
    """Process the mutations file and convert protein mutations to DNA sequences."""
    # Create output file
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['mutant', 'DMS_score', 'sequence'])
        
        # Read the input file
        with open(dms_file, 'r') as infile:
            reader = csv.DictReader(infile)
            
            # Track progress
            total_rows = sum(1 for _ in open(dms_file)) - 1  # Subtract header
            processed = 0
            errors = 0
            
            for row in reader:
                try:
                    protein_mutant = row.get('mutant', '').strip('" ')
                    dms_score = row.get('DMS_score', '')
                    
                    # If no mutation, skip
                    if not protein_mutant:
                        continue
                    
                    # Check if it's a multiple mutation (comma-separated)
                    if ',' in protein_mutant:
                        protein_mutations = protein_mutant.split(',')
                        current_dna = original_dna
                        all_nt_changes = []
                        
                        for mut in protein_mutations:
                            mut = mut.strip('" ')  # Remove quotes and spaces
                            if not mut:  # Skip empty mutations
                                continue
                            
                            orig_aa, pos_1based, new_aa = parse_mutation(mut)
                            pos_0based = pos_1based - 1
                            
                            # Verify the original amino acid matches
                            if 0 <= pos_0based < len(original_protein) and original_protein[pos_0based] != orig_aa:
                                print(f"Mismatch at position {pos_1based}: expected {orig_aa}, found {original_protein[pos_0based]}")
                                raise ValueError(f"Mismatch at position {pos_1based}: expected {orig_aa}, found {original_protein[pos_0based]}")
                            
                            # Apply the mutation to DNA and get nucleotide changes
                            nt_changes, current_dna = apply_mutation_and_get_nucleotide_change(
                                current_dna, pos_1based, new_aa, original_protein
                            )
                            all_nt_changes.extend(nt_changes)
                        
                        # Convert nucleotide changes to comma-separated string
                        nt_mutant = ','.join(all_nt_changes)
                        
                        # Write the output row with nucleotide mutations
                        writer.writerow([nt_mutant, dms_score, current_dna])
                    else:
                        # Single mutation
                        orig_aa, pos_1based, new_aa = parse_mutation(protein_mutant)
                        pos_0based = pos_1based - 1
                        
                        # Verify the original amino acid matches
                        if 0 <= pos_0based < len(original_protein) and original_protein[pos_0based] != orig_aa:
                            print(f"Mismatch at position {pos_1based}: expected {orig_aa}, found {original_protein[pos_0based]}")
                            raise ValueError(f"Mismatch at position {pos_1based}: expected {orig_aa}, found {original_protein[pos_0based]}")
                        
                        # Apply the mutation to DNA and get nucleotide changes
                        nt_changes, mutated_dna = apply_mutation_and_get_nucleotide_change(
                            original_dna, pos_1based, new_aa, original_protein
                        )
                        
                        # Convert nucleotide changes to comma-separated string
                        nt_mutant = ','.join(nt_changes)
                        
                        # Write the output row with nucleotide mutations
                        writer.writerow([nt_mutant, dms_score, mutated_dna])
                    
                except Exception as e:
                    print(f"Error processing mutation {row.get('mutant', 'unknown')}: {e}")
                    # Write the row with ERROR in sequence field
                    writer.writerow([protein_mutant, dms_score, "ERROR"])
                    errors += 1
                
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{total_rows} mutations ({processed/total_rows*100:.1f}%)")
            
            print(f"Completed processing {processed} mutations with {errors} errors.")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Convert protein mutations to nucleotide sequences.')
    parser.add_argument('fasta_file', help='Input FASTA file with nucleotide sequence')
    parser.add_argument('dms_file', help='Input DMS file with mutations')
    parser.add_argument('--output', '-o', default=None, help='Output CSV file (default: input_with_nucleotides.csv)')
    args = parser.parse_args()
    
    # Determine output filename if not specified
    if args.output is None:
        output_file = Path(args.dms_file).stem + '_with_sequences.csv'
    else:
        output_file = args.output
    
    print(f"Reading nucleotide sequence from {args.fasta_file}")
    original_dna = parse_fasta(args.fasta_file)
    print(f"Nucleotide sequence length: {len(original_dna)} bp")
    
    # Translate to protein
    original_protein = translate_dna(original_dna)
    print(f"Translated protein length: {len(original_protein)} amino acids")
    print(f"First 10 amino acids: {original_protein[:10]}")
    print(f"Last 10 amino acids: {original_protein[-10:]}")
    
    # Print the wild-type sequence for reference
    print("\nWild-type nucleotide sequence:")
    print(original_dna)
    
    # Process the mutations file
    print(f"\nProcessing mutations from {args.dms_file}")
    process_mutations_file(args.dms_file, output_file, original_dna, original_protein)
    print(f"Processing complete. Output saved to {output_file}")
    print(original_dna)

if __name__ == "__main__":
    main()
