#!/usr/bin/env python3
"""
Create sample data for testing the experiment_prep pipeline locally.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_dataset(n_sequences=1000, seed=42):
    """Create a sample dataset mimicking BLAST output structure."""
    np.random.seed(seed)
    
    genera = ['Aspergillus', 'Candida', 'Penicillium', 'Fusarium', 'Cryptococcus', 
              'Saccharomyces', 'Trichoderma', 'Alternaria']
    
    data = []
    sequence_counter = 0
    
    for genus in genera:
        n_species = np.random.randint(3, 8)
        
        for sp_idx in range(n_species):
            if sp_idx < n_species - 2:
                species = f"{genus}_species_{sp_idx+1}"
            else:
                species = f"{genus}_sp"
            
            n_seqs = np.random.randint(20, 200)
            
            for _ in range(n_seqs):
                sequence_counter += 1
                
                resolution_level = np.random.choice([4, 5, 6], p=[0.1, 0.3, 0.6])
                
                seq_length = np.random.randint(200, 500)
                sequence = ''.join(np.random.choice(['A', 'T', 'G', 'C'], seq_length))
                
                data.append({
                    'sequence_id': f'seq_{sequence_counter:06d}',
                    'sequence': sequence,
                    'kingdom': 'Fungi',
                    'phylum': f'Phylum_{np.random.randint(1, 4)}',
                    'class': f'Class_{np.random.randint(1, 6)}',
                    'order': f'Order_{np.random.randint(1, 10)}',
                    'family': f'Family_{np.random.randint(1, 15)}',
                    'genus': genus,
                    'species': species,
                    'resolution_level': resolution_level,
                    'confidence_score': np.random.uniform(0.85, 1.0)
                })
    
    df = pd.DataFrame(data)
    
    df = df.sample(n=min(n_sequences, len(df)), random_state=seed).reset_index(drop=True)
    
    return df


def main():
    """Create sample datasets for testing."""
    sample_dir = Path("experiment_prep/sample_data")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        'a_recruited_99pct_90cov_species': 1200,
        'b_recruited_99pct_90cov_sp_conservative': 1000,
        'd_recruited_97pct_80cov_sp_permissive': 800
    }
    
    for dataset_name, n_sequences in datasets.items():
        df = create_sample_dataset(n_sequences)
        output_path = sample_dir / f"{dataset_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(df)} sequences")
    
    print(f"\nSample data created in {sample_dir}")
    print("Run the pipeline with: LOCAL_TEST=1 python scripts/01_assign_folds.py")


if __name__ == "__main__":
    main()