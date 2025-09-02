#!/usr/bin/env python3
"""
02_create_debug_subset.py

Creates a fixed debug subset with 5 diverse genera for quick testing.
Ensures reproducibility by using a fixed seed.
Reads paths from environment and parameters from config.yaml
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_diverse_genera(df: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """
    Select diverse genera for the debug subset.
    Criteria:
    - Has sufficient species (including _sp variants)
    - Represents different taxonomic branches
    - Fixed selection for reproducibility
    """
    debug_config = config['debug_subset']
    n_genera = debug_config['n_genera']
    min_species = debug_config['min_species_per_genus']
    seed = debug_config['seed']
    
    logger.info(f"Selecting {n_genera} genera with at least {min_species} species each")
    
    genus_stats = df.groupby('genus').agg({
        'species': 'nunique',
        'sequence_id': 'count',
        'species_resolution': 'mean'
    }).rename(columns={
        'species': 'n_species',
        'sequence_id': 'n_sequences',
        'species_resolution': 'avg_resolution'
    })
    
    eligible_genera = genus_stats[genus_stats['n_species'] >= min_species]
    logger.info(f"Found {len(eligible_genera)} eligible genera")
    
    if len(eligible_genera) < n_genera:
        logger.warning(f"Only {len(eligible_genera)} genera meet criteria, using all")
        selected_genera = eligible_genera.index.tolist()
    else:
        np.random.seed(seed)
        
        top_by_species = eligible_genera.nlargest(n_genera * 2, 'n_species').index.tolist()
        
        selected_genera = []
        
        selected_genera.append(top_by_species[0])
        
        remaining = [g for g in top_by_species if g != selected_genera[0]]
        np.random.shuffle(remaining)
        selected_genera.extend(remaining[:n_genera-1])
    
    for genus in selected_genera[:n_genera]:
        stats = genus_stats.loc[genus]
        logger.info(f"Selected {genus}: {stats['n_species']} species, "
                   f"{stats['n_sequences']:,} sequences")
    
    return selected_genera[:n_genera]


def create_balanced_subset(df: pd.DataFrame, selected_genera: List[str], 
                          config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a balanced subset from selected genera.
    Maintains fold assignments from the full dataset.
    """
    target_size = config['debug_subset']['target_size']
    seed = config['debug_subset']['seed']
    
    logger.info(f"Creating balanced subset with target size ~{target_size:,}")
    
    subset_dfs = []
    sequences_per_genus = target_size // len(selected_genera)
    
    np.random.seed(seed)
    
    for genus in selected_genera:
        genus_df = df[df['genus'] == genus].copy()
        
        if len(genus_df) <= sequences_per_genus:
            subset_dfs.append(genus_df)
            logger.info(f"{genus}: Using all {len(genus_df)} sequences")
        else:
            species_counts = genus_df['species'].value_counts()
            
            selected_sequences = []
            remaining_budget = sequences_per_genus
            
            for species in species_counts.index:
                if remaining_budget <= 0:
                    break
                
                species_df = genus_df[genus_df['species'] == species]
                n_to_sample = min(len(species_df), max(10, remaining_budget // 10))
                
                if n_to_sample > 0:
                    sampled = species_df.sample(n=min(n_to_sample, len(species_df)), 
                                               random_state=seed)
                    selected_sequences.append(sampled)
                    remaining_budget -= len(sampled)
            
            if remaining_budget > 0 and selected_sequences:
                remaining_df = genus_df[~genus_df.index.isin(
                    pd.concat(selected_sequences).index)]
                if len(remaining_df) > 0:
                    additional = remaining_df.sample(
                        n=min(remaining_budget, len(remaining_df)), 
                        random_state=seed)
                    selected_sequences.append(additional)
            
            genus_subset = pd.concat(selected_sequences)
            subset_dfs.append(genus_subset)
            logger.info(f"{genus}: Sampled {len(genus_subset)} sequences from "
                       f"{genus_subset['species'].nunique()} species")
    
    subset_df = pd.concat(subset_dfs, ignore_index=True)
    
    logger.info(f"Debug subset created with {len(subset_df):,} sequences")
    logger.info(f"Species distribution: {subset_df['species'].nunique()} unique species")
    logger.info(f"Genus distribution:\n{subset_df['genus'].value_counts()}")
    
    return subset_df


def validate_fold_preservation(full_df: pd.DataFrame, subset_df: pd.DataFrame) -> None:
    """Verify that fold assignments are preserved from full dataset."""
    
    merged = subset_df.merge(
        full_df[['sequence_id', 'fold_exp1', 'fold_exp2']], 
        on='sequence_id', 
        suffixes=('_subset', '_full')
    )
    
    exp1_match = (merged['fold_exp1_subset'] == merged['fold_exp1_full']).all()
    exp2_match = (merged['fold_exp2_subset'] == merged['fold_exp2_full']).all()
    
    if exp1_match and exp2_match:
        logger.info("✓ Fold assignments preserved correctly")
    else:
        logger.error("✗ Fold assignments do not match!")
        if not exp1_match:
            logger.error("  - exp1 folds mismatch")
        if not exp2_match:
            logger.error("  - exp2 folds mismatch")


def process_union(output_dir: Path, config: Dict[str, Any], union_name: str) -> None:
    """Process a union dataset to create debug subset."""
    
    logger.info(f"Creating debug subset for union: {union_name}")
    
    for exp_name in ['exp1_sequence_fold', 'exp2_species_fold']:
        full_path = output_dir / exp_name / 'full_10fold' / f"{union_name}.csv"
        
        if not full_path.exists():
            logger.warning(f"Full union dataset not found: {full_path}")
            logger.warning("Run 01_assign_folds.py first!")
            continue
        
        df = pd.read_csv(full_path)
        logger.info(f"Loaded {len(df):,} sequences from {exp_name}")
        
        selected_genera = select_diverse_genera(df, config)
        
        subset_df = create_balanced_subset(df, selected_genera, config)
        
        validate_fold_preservation(df, subset_df)
        
        debug_dir = output_dir / exp_name / 'debug_5genera_10fold'
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = debug_dir / f"{union_name}.csv"
        subset_df.to_csv(output_path, index=False)
        logger.info(f"Saved debug subset to {output_path}")
        
        fold_dist = subset_df['fold_exp1' if 'exp1' in exp_name else 'fold_exp2'].value_counts().sort_index()
        logger.info(f"Fold distribution in debug subset:\n{fold_dist}")


def main():
    """Main execution function."""
    # Load environment variables from central .env
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(f"{env_path} not found. Using system environment variables.")
    
    # Load operational config from YAML
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    
    if not config['debug_subset']['enabled']:
        logger.info("Debug subset creation disabled in config")
        return
    
    # Get paths from environment
    if 'LOCAL_TEST' in os.environ:
        logger.info("Running in LOCAL_TEST mode")
        output_dir = Path("../experiment_prep/output")
    else:
        experiments_dir = os.getenv('EXPERIMENTS_DIR')
        if not experiments_dir:
            raise ValueError("EXPERIMENTS_DIR must be set in .env")
        output_dir = Path(experiments_dir)
    
    # Process the union datasets
    unions = ['standard', 'conservative']
    
    for union_name in unions:
        process_union(output_dir, config, union_name)
    
    logger.info("Debug subset creation complete!")


if __name__ == "__main__":
    main()