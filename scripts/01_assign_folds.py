#!/usr/bin/env python3
"""
01_assign_folds.py

Assigns k-fold cross-validation splits to sequences for two experiment types:
- exp1_sequence_fold: Standard stratified k-fold by resolution level
- exp2_species_fold: Group k-fold where all sequences of a species stay together

Reads paths from environment and parameters from config.yaml
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from typing import Dict, Any, List
import logging
from dotenv import load_dotenv

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


def create_dataset_union(input_dir: Path, dataset_files: List[str], union_name: str) -> pd.DataFrame:
    """Create a union of multiple dataset files."""
    logger.info(f"Creating union '{union_name}' from files: {dataset_files}")
    
    dfs = []
    for dataset_file in dataset_files:
        file_path = input_dir / f"{dataset_file}.csv"
        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(f"Required dataset {dataset_file} not found")
        
        df = pd.read_csv(file_path)
        logger.info(f"  - {dataset_file}: {len(df):,} sequences")
        dfs.append(df)
    
    union_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on sequence_id
    initial_count = len(union_df)
    union_df = union_df.drop_duplicates(subset=['sequence_id'], keep='first')
    final_count = len(union_df)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate sequences")
    
    logger.info(f"Union '{union_name}' created with {len(union_df):,} sequences")
    return union_df


def process_union(union_df: pd.DataFrame, output_dir: Path, config: Dict[str, Any], 
                  union_name: str) -> None:
    """Process a dataset union - filter and assign folds."""
    logger.info(f"Processing union: {union_name}")
    logger.info(f"Initial size: {len(union_df):,} sequences")
    
    df = filter_by_resolution(union_df)
    
    df = assign_sequence_folds(df, config)
    df = assign_species_folds(df, config)
    
    for exp_name in ['exp1_sequence_fold', 'exp2_species_fold']:
        exp_dir = output_dir / exp_name / 'full_10fold'
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = exp_dir / f"{union_name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")


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
    
    # Get paths from environment
    if 'LOCAL_TEST' in os.environ:
        logger.info("Running in LOCAL_TEST mode with sample data")
        input_dir = Path("../experiment_prep/sample_data")
        output_dir = Path("../experiment_prep/output")
    else:
        # Use environment variables for paths
        blast_filtered_dir = os.getenv('BLAST_FILTERED_DIR')
        experiments_dir = os.getenv('EXPERIMENTS_DIR')
        
        if not blast_filtered_dir or not experiments_dir:
            raise ValueError("BLAST_FILTERED_DIR and EXPERIMENTS_DIR must be set in .env")
            
        input_dir = Path(blast_filtered_dir)
        output_dir = Path(experiments_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the experimental setups based on BLAST output:
    # b = non-_sp sequences at 99% identity (species-level, high confidence)
    # c = _sp sequences at 99% identity (conservative approach for uncertain taxa)
    # d = _sp sequences at 97% identity (permissive approach for uncertain taxa)
    
    # Standard/default approach: use permissive threshold for _sp
    unions = {
        'standard': ['b_recruited_99pct_species', 'd_recruited_97pct_sp'],      # b+d (default)
        'conservative': ['b_recruited_99pct_species', 'c_recruited_99pct_sp']   # b+c (optional)
    }
    
    logger.info("Creating dataset unions:")
    logger.info("  - standard (b+d): non-_sp at 99% + _sp at 97% [DEFAULT]")
    logger.info("  - conservative (b+c): non-_sp at 99% + _sp at 99% [OPTIONAL]")
    
    for union_name, dataset_files in unions.items():
        try:
            union_df = create_dataset_union(input_dir, dataset_files, union_name)
            process_union(union_df, output_dir, config, union_name)
        except FileNotFoundError as e:
            logger.error(f"Skipping {union_name} union: {e}")
            continue
    
    logger.info("Fold assignment complete!")


if __name__ == "__main__":
    main()