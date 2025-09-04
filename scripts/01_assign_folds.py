#!/usr/bin/env python3
"""
01_assign_folds.py

Assigns k-fold cross-validation splits to sequences for two experiment types:
- exp1_sequence_fold: Standard stratified k-fold by species labels (updated)
- exp2_species_fold: Group k-fold where all sequences of a species stay together

Now includes sequence-content deduplication within species after filtering.
Reads paths from environment and parameters from config.yaml
"""

import pandas as pd
import numpy as np
import yaml
import os
import json
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


class LabelEncoder:
    """Label encoder that returns None for unknown labels - same as granular_control"""
    
    def __init__(self, labels):
        """Initialize encoder from list of labels"""
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    def encode(self, label):
        """Encode label to index, returns None for unknown"""
        return self.label_to_index.get(label, None)
    
    def decode(self, index):
        """Decode index to label"""
        return self.index_to_label.get(index, None)
    
    def to_dict(self):
        """Convert to JSON-serializable dictionary"""
        return {
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()}
        }


def save_fold_label_encoders(df: pd.DataFrame, output_dir: Path, union_name: str, 
                             fold_column: str = 'fold_exp1') -> None:
    """Save separate label encoders for each fold - built from training data only"""
    
    logger.info(f"Creating per-fold label encoders for {union_name}")
    taxonomic_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    
    for fold in range(1, 11):
        # Build encoders from TRAINING folds only (exclude validation fold)
        train_df = df[df[fold_column] != fold]
        
        encoders = {}
        for level in taxonomic_levels:
            # Get unique labels from TRAINING data only
            train_labels = train_df[level].dropna().tolist()
            
            # Build encoder
            encoder = LabelEncoder(train_labels)
            encoder_dict = encoder.to_dict()
            encoder_dict['num_classes'] = len(encoder.label_to_index)
            
            # Add stats about what's missing in validation
            val_df = df[df[fold_column] == fold]
            val_labels = set(val_df[level].dropna().unique())
            train_labels_set = set(encoder.label_to_index.keys())
            unknown_labels = val_labels - train_labels_set
            
            encoder_dict['num_unknown_in_val'] = len(unknown_labels)
            if unknown_labels:
                encoder_dict['example_unknown'] = list(unknown_labels)[:5]
            
            encoders[level] = encoder_dict
        
        # Save fold-specific encoder
        encoder_path = output_dir / f"label_encoders_{union_name}_fold{fold}.json"
        with open(encoder_path, 'w') as f:
            json.dump(encoders, f, indent=2)
    
    logger.info(f"Saved 10 fold-specific label encoders for {union_name}")


def filter_by_resolution(df: pd.DataFrame, min_resolution: int = 4) -> pd.DataFrame:
    """Filter sequences by minimum species resolution level."""
    logger.info(f"Filtering sequences with species_resolution > {min_resolution}")
    initial_count = len(df)
    df_filtered = df[df['species_resolution'] > min_resolution].copy()
    final_count = len(df_filtered)
    logger.info(f"Filtered from {initial_count:,} to {final_count:,} sequences")
    return df_filtered


def deduplicate_within_species(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate sequences (by content) within each species.
    Keeps duplicate sequences across different species.
    
    Args:
        df: DataFrame with 'species' and 'sequence' columns
        keep: Which duplicate to keep ('first', 'last', or 'resolution')
              'resolution' keeps the one with highest species_resolution
    
    Returns:
        DataFrame with duplicates removed within each species
    """
    logger.info("="*60)
    logger.info("DEDUPLICATING SEQUENCES WITHIN SPECIES")
    logger.info("="*60)
    
    initial_count = len(df)
    initial_species = df['species'].nunique()
    
    # Calculate initial statistics
    logger.info(f"Initial dataset:")
    logger.info(f"  Total sequences: {initial_count:,}")
    logger.info(f"  Unique species: {initial_species:,}")
    logger.info(f"  Unique sequences overall: {df['sequence'].nunique():,}")
    
    # Track deduplication statistics per species
    dedup_stats = []
    
    if keep == 'resolution':
        # Sort by species and resolution (descending) to keep highest resolution
        df_sorted = df.sort_values(['species', 'species_resolution'], 
                                   ascending=[True, False])
        # Group by species and remove duplicates within each group
        df_dedup = df_sorted.groupby('species', group_keys=False).apply(
            lambda x: x.drop_duplicates(subset=['sequence'], keep='first')
        ).reset_index(drop=True)
    else:
        # Simple deduplication keeping first or last
        df_dedup = df.groupby('species', group_keys=False).apply(
            lambda x: x.drop_duplicates(subset=['sequence'], keep=keep)
        ).reset_index(drop=True)
    
    # Calculate deduplication statistics
    final_count = len(df_dedup)
    sequences_removed = initial_count - final_count
    reduction_pct = (sequences_removed / initial_count) * 100
    
    logger.info(f"\nDeduplication results:")
    logger.info(f"  Sequences removed: {sequences_removed:,} ({reduction_pct:.1f}%)")
    logger.info(f"  Sequences remaining: {final_count:,}")
    logger.info(f"  Unique species preserved: {df_dedup['species'].nunique():,}")
    
    # Show top species by deduplication
    species_counts_before = df.groupby('species').size()
    species_counts_after = df_dedup.groupby('species').size()
    
    dedup_by_species = pd.DataFrame({
        'before': species_counts_before,
        'after': species_counts_after
    }).fillna(0)
    dedup_by_species['removed'] = dedup_by_species['before'] - dedup_by_species['after']
    dedup_by_species['removal_pct'] = (dedup_by_species['removed'] / dedup_by_species['before']) * 100
    
    logger.info(f"\nTop 10 species by sequences removed:")
    top_dedup = dedup_by_species.nlargest(10, 'removed')
    for species, row in top_dedup.iterrows():
        logger.info(f"  {species[:40]:<40} {int(row['before']):>6} → {int(row['after']):>6} "
                   f"(-{int(row['removed']):>5}, {row['removal_pct']:>5.1f}%)")
    
    # Check for cross-species duplicates (informative)
    logger.info(f"\nCross-species sequence sharing:")
    sequence_species_counts = df_dedup.groupby('sequence')['species'].nunique()
    shared_sequences = sequence_species_counts[sequence_species_counts > 1]
    if len(shared_sequences) > 0:
        logger.info(f"  {len(shared_sequences):,} sequences appear in multiple species")
        logger.info(f"  Max species sharing one sequence: {shared_sequences.max()}")
        
        # Show examples of highly shared sequences
        most_shared = shared_sequences.nlargest(5)
        logger.info(f"  Top shared sequences (appearing in N species):")
        for seq_hash, n_species in most_shared.items():
            # Get the species that share this sequence
            species_list = df_dedup[df_dedup['sequence'] == seq_hash]['species'].unique()[:3]
            species_str = ', '.join(species_list[:3])
            if len(species_list) > 3:
                species_str += f", ... ({n_species} total)"
            logger.info(f"    Sequence appears in {n_species} species: {species_str}")
    else:
        logger.info(f"  No sequences are shared between species")
    
    logger.info("="*60)
    
    return df_dedup


def assign_sequence_folds(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assign standard k-fold splits stratified by species labels.
    Each sequence is independently assigned to a fold.
    Updated to stratify by species instead of resolution.
    """
    exp_config = config['experiments']['exp1_sequence_fold']
    k_folds = exp_config['k_folds']
    # Changed default from 'species_resolution' to 'species'
    stratify_by = exp_config.get('stratify_by', 'species')
    seed = exp_config['seed']
    
    logger.info(f"Assigning sequence-level {k_folds}-fold splits, stratified by {stratify_by}")
    
    # Check if we're stratifying by species (new behavior)
    if stratify_by == 'species':
        # Count sequences per species
        species_counts = df['species'].value_counts()
        rare_species = species_counts[species_counts < k_folds]
        
        if len(rare_species) > 0:
            logger.warning(f"Warning: {len(rare_species):,} species have fewer than {k_folds} sequences")
            logger.warning(f"These species will not appear in all folds")
            logger.info(f"Species with <{k_folds} sequences represent "
                       f"{df['species'].isin(rare_species.index).sum():,} sequences "
                       f"({df['species'].isin(rare_species.index).sum()/len(df)*100:.1f}% of data)")
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    df['fold_exp1'] = -1
    
    X = np.arange(len(df))
    y = df[stratify_by].values
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        df.iloc[val_idx, df.columns.get_loc('fold_exp1')] = fold_idx
    
    fold_counts = df['fold_exp1'].value_counts().sort_index()
    logger.info(f"Fold distribution for exp1:\n{fold_counts}")
    
    # If stratifying by species, show species distribution
    if stratify_by == 'species':
        species_per_fold = df.groupby('fold_exp1')['species'].nunique()
        logger.info(f"Unique species per fold:\n{species_per_fold}")
        
        # Check fold balance for common species
        common_species = species_counts[species_counts >= k_folds].index[:10]
        logger.info(f"\nFold distribution for top 10 common species:")
        for species in common_species:
            species_df = df[df['species'] == species]
            fold_dist = species_df['fold_exp1'].value_counts().sort_index()
            logger.info(f"  {species[:30]:<30} {list(fold_dist.values)}")
    
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
    
    # Remove duplicates based on sequence_id (keep technical duplicates out)
    initial_count = len(union_df)
    union_df = union_df.drop_duplicates(subset=['sequence_id'], keep='first')
    final_count = len(union_df)
    
    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate sequence_ids")
    
    logger.info(f"Union '{union_name}' created with {len(union_df):,} sequences")
    return union_df


def process_union(union_df: pd.DataFrame, output_dir: Path, config: Dict[str, Any], 
                  union_name: str) -> None:
    """Process a dataset union - filter, deduplicate, and assign folds."""
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING UNION: {union_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Initial size: {len(union_df):,} sequences")
    
    # Step 1: Filter by resolution (keep good quality annotations)
    df = filter_by_resolution(union_df)
    
    # Step 2: Deduplicate within species (remove sequence-content duplicates)
    # Use 'resolution' to keep highest resolution version when deduplicating
    df = deduplicate_within_species(df, keep='resolution')
    
    # Step 3: Assign folds for both experiments
    df = assign_sequence_folds(df, config)
    df = assign_species_folds(df, config)
    
    # Save the processed data and label encoders
    for exp_name in ['exp1_sequence_fold', 'exp2_species_fold']:
        exp_dir = output_dir / exp_name / 'full_10fold' / 'data'
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = exp_dir / f"{union_name}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        
        # Save fold-specific label encoders
        fold_column = 'fold_exp1' if 'exp1' in exp_name else 'fold_exp2'
        save_fold_label_encoders(df, exp_dir, union_name, fold_column)
    
    # Also save for debug subsets
    for exp_name in ['exp1_sequence_fold', 'exp2_species_fold']:
        debug_dir = output_dir / exp_name / 'debug_5genera_10fold' / 'data'
        if debug_dir.exists():
            # Check if debug subset exists
            debug_csv = debug_dir / f"{union_name}.csv"
            if debug_csv.exists():
                debug_df = pd.read_csv(debug_csv)
                fold_column = 'fold_exp1' if 'exp1' in exp_name else 'fold_exp2'
                save_fold_label_encoders(debug_df, debug_dir, union_name, fold_column)
                logger.info(f"Saved label encoders for debug subset: {exp_name}/{union_name}")
    
    logger.info(f"{'='*70}")
    logger.info(f"COMPLETED: {union_name}")
    logger.info(f"Final size: {len(df):,} sequences")
    logger.info(f"{'='*70}\n")


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
    # a = original SH sequences (reference sequences)
    # b = recruited non-_sp sequences at 99% identity (species-level, high confidence)
    # c = recruited _sp sequences at 99% identity (conservative approach for uncertain taxa)
    # d = recruited _sp sequences at 97% identity (permissive approach for uncertain taxa)
    
    # Unions include original sequences (a) plus recruited sequences
    unions = {
        'standard': ['a_original_sh_sequences', 'b_recruited_99pct_species', 'd_recruited_97pct_sp'],      # a+b+d (default)
        'conservative': ['a_original_sh_sequences', 'b_recruited_99pct_species', 'c_recruited_99pct_sp']   # a+b+c (optional)
    }
    
    logger.info("="*70)
    logger.info("EXPERIMENT PREPARATION WITH DEDUPLICATION")
    logger.info("="*70)
    logger.info("Creating dataset unions with sequence-content deduplication:")
    logger.info("  - standard (a+b+d): original + non-_sp at 99% + _sp at 97% [DEFAULT]")
    logger.info("  - conservative (a+b+c): original + non-_sp at 99% + _sp at 99% [OPTIONAL]")
    logger.info("\nNew features:")
    logger.info("  ✓ Sequence-content deduplication within species")
    logger.info("  ✓ Species-label stratification for exp1")
    logger.info("="*70 + "\n")
    
    for union_name, dataset_files in unions.items():
        try:
            union_df = create_dataset_union(input_dir, dataset_files, union_name)
            process_union(union_df, output_dir, config, union_name)
        except FileNotFoundError as e:
            logger.error(f"Skipping {union_name} union: {e}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("FOLD ASSIGNMENT COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()