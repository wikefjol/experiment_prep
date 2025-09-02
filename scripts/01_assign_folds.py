#!/usr/bin/env python3
"""
01_assign_folds.py

Assigns k-fold cross-validation splits to sequences for two experiment types:
- exp1_sequence_fold: Standard stratified k-fold by resolution level
- exp2_species_fold: Group k-fold where all sequences of a species stay together
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def filter_by_resolution(df: pd.DataFrame, min_resolution: int = 4) -> pd.DataFrame:
    """Filter sequences by minimum species resolution level."""
    logger.info(f"Filtering sequences with species_resolution > {min_resolution}")
    initial_count = len(df)
    df_filtered = df[df['species_resolution'] > min_resolution].copy()
    final_count = len(df_filtered)
    logger.info(f"Filtered from {initial_count:,} to {final_count:,} sequences")
    return df_filtered


def assign_sequence_folds(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assign standard k-fold splits stratified by resolution level.
    Each sequence is independently assigned to a fold.
    """
    exp_config = config['experiments']['exp1_sequence_fold']
    k_folds = exp_config['k_folds']
    stratify_by = exp_config.get('stratify_by', 'species_resolution')  # Use species_resolution as default
    seed = exp_config['seed']
    
    logger.info(f"Assigning sequence-level {k_folds}-fold splits, stratified by {stratify_by}")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    df['fold_exp1'] = -1
    
    X = np.arange(len(df))
    y = df[stratify_by].values
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        df.iloc[val_idx, df.columns.get_loc('fold_exp1')] = fold_idx
    
    fold_counts = df['fold_exp1'].value_counts().sort_index()
    logger.info(f"Fold distribution for exp1:\n{fold_counts}")
    
    return df


def assign_species_folds(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assign group k-fold splits where all sequences of the same species 
    are assigned to the same fold. Stratified by genus when possible.
    """
    exp_config = config['experiments']['exp2_species_fold']
    k_folds = exp_config['k_folds']
    stratify_by = exp_config.get('stratify_by', 'genus')  # Keep genus for species grouping
    seed = exp_config['seed']
    
    logger.info(f"Assigning species-level {k_folds}-fold splits, stratified by {stratify_by}")
    
    species_to_genus = df.groupby('species')[stratify_by].first()
    unique_species = species_to_genus.index.values
    species_genera = species_to_genus.values
    
    logger.info(f"Found {len(unique_species)} unique species")
    
    # Check if we have enough species per genus for stratification
    genus_counts = pd.Series(species_genera).value_counts()
    min_genus_count = genus_counts.min()
    
    if len(unique_species) < k_folds or min_genus_count < k_folds:
        logger.warning(f"Only {len(unique_species)} species or insufficient genus samples, using regular split")
        # Use simple random split without stratification
        np.random.seed(seed)
        shuffled_species = unique_species.copy()
        np.random.shuffle(shuffled_species)
        species_folds = np.array_split(shuffled_species, k_folds)
        species_to_fold = {}
        for fold_idx, species_list in enumerate(species_folds, 1):
            for species in species_list:
                species_to_fold[species] = fold_idx
    else:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
        species_to_fold = {}
        X = np.arange(len(unique_species))
        y = species_genera
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            for idx in val_idx:
                species_to_fold[unique_species[idx]] = fold_idx
    
    df['fold_exp2'] = df['species'].map(species_to_fold)
    
    fold_counts = df['fold_exp2'].value_counts().sort_index()
    logger.info(f"Fold distribution for exp2:\n{fold_counts}")
    
    species_per_fold = df.groupby('fold_exp2')['species'].nunique()
    logger.info(f"Species per fold in exp2:\n{species_per_fold}")
    
    return df


def process_dataset(input_path: Path, output_dir: Path, config: Dict[str, Any], 
                   dataset_name: str) -> None:
    """Process a single dataset file."""
    logger.info(f"Processing {dataset_name} from {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} sequences")
    
    df = filter_by_resolution(df)
    
    df = assign_sequence_folds(df, config)
    df = assign_species_folds(df, config)
    
    for exp_name in ['exp1_sequence_fold', 'exp2_species_fold']:
        exp_dir = output_dir / exp_name / 'full_10fold'
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = exp_dir / f"{dataset_name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")


def main():
    """Main execution function."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    
    if 'LOCAL_TEST' in os.environ:
        logger.info("Running in LOCAL_TEST mode with sample data")
        input_dir = Path("../experiment_prep/sample_data")
        output_dir = Path("../experiment_prep/output")
    else:
        input_dir = Path(config['input']['blast_output'])
        output_dir = Path(config['output']['base_path'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        'a_recruited_99pct_90cov_species',
        'b_recruited_99pct_90cov_sp_conservative',
        'd_recruited_97pct_80cov_sp_permissive'
    ]
    
    for dataset_name in datasets:
        input_path = input_dir / f"{dataset_name}.csv"
        
        if not input_path.exists():
            logger.warning(f"Dataset not found: {input_path}")
            continue
        
        process_dataset(input_path, output_dir, config, dataset_name)
    
    logger.info("Fold assignment complete!")


if __name__ == "__main__":
    main()