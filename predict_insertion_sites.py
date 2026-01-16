#!/usr/bin/env python3
"""
ProDomino - Predict insertion sites for a protein sequence.

Usage examples:
    # Using GRK2 sequence from UniProt
    python predict_insertion_sites.py --uniprot P21146 --output results/grk2

    # Using a FASTA file
    python predict_insertion_sites.py --fasta my_protein.fasta --output results/my_protein

    # Using a raw sequence
    python predict_insertion_sites.py --sequence "MSEQ..." --output results/output

    # With PDB structure for B-factor mapping
    python predict_insertion_sites.py --fasta grk2.fasta --pdb grk2.pdb --chain A --output results/grk2
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from ProDomino import Embedder, ProDomino


def parse_fasta(fasta_path):
    """Parse a FASTA file and return the first sequence."""
    sequences = {}
    current_header = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = ''.join(current_seq)
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header is not None:
            sequences[current_header] = ''.join(current_seq)

    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")

    # Return first sequence
    name = list(sequences.keys())[0]
    return name, sequences[name]


def fetch_uniprot_sequence(uniprot_id):
    """Fetch sequence from UniProt."""
    import urllib.request

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"Fetching sequence from UniProt: {url}")

    try:
        with urllib.request.urlopen(url) as response:
            fasta_content = response.read().decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to fetch UniProt sequence: {e}")

    lines = fasta_content.strip().split('\n')
    header = lines[0][1:]
    sequence = ''.join(lines[1:])

    return header.split()[0], sequence


def save_results(prediction, output_dir, name, sequence, pdb_path=None, chain_id='A', shift=0):
    """Save prediction results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save raw predictions as numpy array
    predictions_array = prediction.predicted_sites.numpy()
    np.save(os.path.join(output_dir, f'{name}_predictions.npy'), predictions_array)
    print(f"Saved predictions to {output_dir}/{name}_predictions.npy")

    # Save as CSV with residue positions
    csv_path = os.path.join(output_dir, f'{name}_predictions.csv')
    with open(csv_path, 'w') as f:
        f.write("position,residue,prediction_score\n")
        for i, (aa, score) in enumerate(zip(sequence, predictions_array)):
            f.write(f"{i+1},{aa},{score:.4f}\n")
    print(f"Saved CSV to {csv_path}")

    # Save top hits
    top_hits = prediction.get_top_hits(n=20)
    top_hits_path = os.path.join(output_dir, f'{name}_top_hits.txt')
    with open(top_hits_path, 'w') as f:
        f.write("# Top 20 predicted insertion sites\n")
        f.write("# Position (1-indexed), Residue, Score\n")
        for pos in sorted(top_hits, key=lambda x: -predictions_array[x]):
            f.write(f"{pos+1}\t{sequence[pos]}\t{predictions_array[pos]:.4f}\n")
    print(f"Saved top hits to {top_hits_path}")

    # If PDB provided, save structure with B-factors
    if pdb_path is not None:
        from Bio.PDB import PDBIO

        prediction.add_sequence(sequence)
        prediction.add_pdb_file(pdb_path, chain_id=chain_id, shift=shift)
        prediction.generate_insertion_site_pdb_file()

        pdbio = PDBIO()
        pdbio.set_structure(prediction.pdb)
        pdb_output = os.path.join(output_dir, f'{name}_bfactor.pdb')
        pdbio.save(pdb_output)
        print(f"Saved PDB with B-factor scores to {pdb_output}")


def main():
    parser = argparse.ArgumentParser(
        description='ProDomino - Predict domain insertion sites in proteins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--fasta', type=str, help='Path to FASTA file')
    input_group.add_argument('--sequence', type=str, help='Raw amino acid sequence')
    input_group.add_argument('--uniprot', type=str, help='UniProt ID to fetch (e.g., P21146 for human GRK2)')

    # Output options
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--name', type=str, default=None,
                        help='Name for output files (default: derived from input)')

    # Model options
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/main_checkpoint.ckpt',
                        help='Path to model checkpoint (default: checkpoints/main_checkpoint.ckpt)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto-detect)')

    # PDB options
    parser.add_argument('--pdb', type=str, default=None,
                        help='Optional PDB file for structure mapping')
    parser.add_argument('--chain', type=str, default='A',
                        help='Chain ID in PDB file (default: A)')
    parser.add_argument('--shift', type=int, default=0,
                        help='Shift for PDB residue numbering (default: 0)')

    args = parser.parse_args()

    # Get sequence
    print("=" * 60)
    print("ProDomino - Domain Insertion Site Prediction")
    print("=" * 60)

    if args.fasta:
        name, sequence = parse_fasta(args.fasta)
        print(f"Loaded sequence from FASTA: {args.fasta}")
    elif args.uniprot:
        name, sequence = fetch_uniprot_sequence(args.uniprot)
        print(f"Fetched sequence from UniProt: {args.uniprot}")
    else:
        name = "sequence"
        sequence = args.sequence
        print("Using provided sequence")

    # Override name if specified
    if args.name:
        name = args.name

    print(f"Sequence name: {name}")
    print(f"Sequence length: {len(sequence)} residues")
    print()

    # Initialize embedder (loads ESM-2 model)
    print("Initializing ESM-2 embedder...")
    embedder = Embedder()
    print("ESM-2 model loaded successfully")
    print()

    # Generate embeddings
    print("Generating ESM-2 embeddings...")
    embedding = embedder.predict_embedding(sequence, name=name)
    print(f"Embedding shape: {embedding.shape}")
    print()

    # Load ProDomino model
    print(f"Loading ProDomino model from {args.checkpoint}...")
    model = ProDomino(args.checkpoint, 'mini_3b_mlp', device=args.device)
    print("ProDomino model loaded successfully")
    print()

    # Predict insertion sites
    print("Predicting insertion sites...")
    prediction = model.predict_insertion_sites(embedding)
    prediction.add_sequence(sequence)
    print("Prediction complete!")
    print()

    # Show summary
    predictions = prediction.predicted_sites.numpy()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Mean prediction score: {predictions.mean():.4f}")
    print(f"Max prediction score:  {predictions.max():.4f}")
    print(f"Min prediction score:  {predictions.min():.4f}")
    print()

    top_hits = prediction.get_top_hits(n=10)
    print("Top 10 predicted insertion sites:")
    for pos in sorted(top_hits, key=lambda x: -predictions[x]):
        print(f"  Position {pos+1:4d} ({sequence[pos]}): {predictions[pos]:.4f}")
    print()

    # Save results
    print("Saving results...")
    save_results(
        prediction=prediction,
        output_dir=args.output,
        name=name,
        sequence=sequence,
        pdb_path=args.pdb,
        chain_id=args.chain,
        shift=args.shift
    )
    print()
    print("Done!")


if __name__ == '__main__':
    main()
