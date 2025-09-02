#!/usr/bin/env python3
"""
03_validate_folds.py

Validates k-fold assignments for both experiment types.
Checks balance, species grouping, and generates statistics.
Reads paths from environment and parameters from config.yaml
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
import logging
from collections import defaultdict
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "../config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_fold_balance(df: pd.DataFrame, fold_col: str, tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate that folds are balanced in size.
    
    Args:
        df: DataFrame with fold assignments
        fold_col: Name of fold column to validate
        tolerance: Maximum allowed deviation from mean (as fraction)
    
    Returns:
        Dictionary with validation results
    """
    fold_sizes = df[fold_col].value_counts().sort_index()
    mean_size = fold_sizes.mean()
    std_size = fold_sizes.std()
    
    max_deviation = (fold_sizes.max() - mean_size) / mean_size
    min_deviation = (mean_size - fold_sizes.min()) / mean_size
    
    is_balanced = max(max_deviation, min_deviation) <= tolerance
    
    results = {
        'fold_sizes': fold_sizes.to_dict(),
        'mean_size': mean_size,
        'std_size': std_size,
        'max_deviation_pct': max_deviation * 100,
        'min_deviation_pct': min_deviation * 100,
        'is_balanced': is_balanced
    }
    
    logger.info(f"Fold balance for {fold_col}:")
    logger.info(f"  Mean size: {mean_size:.1f} ± {std_size:.1f}")
    logger.info(f"  Max deviation: {max_deviation:.1%}")
    logger.info(f"  Status: {'✓ Balanced' if is_balanced else '✗ Imbalanced'}")
    
    return results


def validate_stratification(df: pd.DataFrame, fold_col: str, stratify_col: str) -> Dict[str, Any]:
    """
    Validate that stratification is maintained across folds.
    """
    overall_dist = df[stratify_col].value_counts(normalize=True)
    
    fold_distributions = {}
    max_deviation = 0
    
    for fold in sorted(df[fold_col].unique()):
        fold_df = df[df[fold_col] == fold]
        fold_dist = fold_df[stratify_col].value_counts(normalize=True)
        
        deviations = []
        for level in overall_dist.index:
            if level in fold_dist.index:
                dev = abs(fold_dist[level] - overall_dist[level])
                deviations.append(dev)
                max_deviation = max(max_deviation, dev)
        
        fold_distributions[f'fold_{fold}'] = {
            'distribution': fold_dist.to_dict(),
            'mean_deviation': np.mean(deviations) if deviations else 0
        }
    
    results = {
        'overall_distribution': overall_dist.to_dict(),
        'fold_distributions': fold_distributions,
        'max_deviation': max_deviation,
        'is_well_stratified': max_deviation < 0.05
    }
    
    logger.info(f"Stratification by {stratify_col}:")
    logger.info(f"  Max deviation from overall: {max_deviation:.1%}")
    logger.info(f"  Status: {'✓ Well stratified' if results['is_well_stratified'] else '⚠ Some imbalance'}")
    
    return results


def validate_species_grouping(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that all sequences of a species are in the same fold for exp2.
    """
    species_folds = df.groupby('species')['fold_exp2'].nunique()
    
    violations = species_folds[species_folds > 1]
    
    results = {
        'total_species': len(species_folds),
        'species_with_violations': len(violations),
        'violation_species': violations.index.tolist() if len(violations) > 0 else [],
        'is_valid': len(violations) == 0
    }
    
    logger.info("Species grouping validation for exp2:")
    if results['is_valid']:
        logger.info("  ✓ All species correctly grouped")
    else:
        logger.error(f"  ✗ {len(violations)} species split across multiple folds!")
        for species in violations.index[:5]:
            folds = df[df['species'] == species]['fold_exp2'].unique()
            logger.error(f"    - {species}: in folds {sorted(folds)}")
    
    return results


def validate_genus_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check how genera are distributed across folds.
    """
    results = {}
    
    for fold_col in ['fold_exp1', 'fold_exp2']:
        genus_fold_matrix = pd.crosstab(df['genus'], df[fold_col])
        
        genera_per_fold = (genus_fold_matrix > 0).sum(axis=0)
        folds_per_genus = (genus_fold_matrix > 0).sum(axis=1)
        
        results[fold_col] = {
            'genera_per_fold': genera_per_fold.to_dict(),
            'mean_genera_per_fold': genera_per_fold.mean(),
            'mean_folds_per_genus': folds_per_genus.mean(),
            'genera_in_all_folds': (folds_per_genus == df[fold_col].nunique()).sum(),
            'genera_in_one_fold': (folds_per_genus == 1).sum()
        }
        
        logger.info(f"Genus distribution for {fold_col}:")
        logger.info(f"  Mean genera per fold: {genera_per_fold.mean():.1f}")
        logger.info(f"  Genera in all folds: {results[fold_col]['genera_in_all_folds']}")
        logger.info(f"  Genera in only one fold: {results[fold_col]['genera_in_one_fold']}")
    
    return results


def generate_summary_report(df: pd.DataFrame, output_path: Path, dataset_name: str) -> None:
    """Generate a comprehensive validation report."""
    
    report_lines = [
        f"# Fold Validation Report: {dataset_name}",
        f"",
        f"## Dataset Overview",
        f"- Total sequences: {len(df):,}",
        f"- Unique species: {df['species'].nunique():,}",
        f"- Unique genera: {df['genus'].nunique():,}",
        f"- Resolution levels: {df['species_resolution'].value_counts().sort_index().to_dict()}",
        f""
    ]
    
    exp1_balance = validate_fold_balance(df, 'fold_exp1')
    report_lines.extend([
        f"## Experiment 1: Sequence-level Folds",
        f"- Fold sizes: {exp1_balance['fold_sizes']}",
        f"- Balance status: {'✓' if exp1_balance['is_balanced'] else '✗'}",
        f""
    ])
    
    exp1_strat = validate_stratification(df, 'fold_exp1', 'species_resolution')
    report_lines.extend([
        f"### Stratification by Resolution Level",
        f"- Max deviation: {exp1_strat['max_deviation']:.1%}",
        f"- Status: {'✓' if exp1_strat['is_well_stratified'] else '⚠'}",
        f""
    ])
    
    exp2_balance = validate_fold_balance(df, 'fold_exp2')
    report_lines.extend([
        f"## Experiment 2: Species-level Folds",
        f"- Fold sizes: {exp2_balance['fold_sizes']}",
        f"- Balance status: {'✓' if exp2_balance['is_balanced'] else '✗'}",
        f""
    ])
    
    species_grouping = validate_species_grouping(df)
    report_lines.extend([
        f"### Species Grouping",
        f"- Total species: {species_grouping['total_species']}",
        f"- Violations: {species_grouping['species_with_violations']}",
        f"- Status: {'✓' if species_grouping['is_valid'] else '✗'}",
        f""
    ])
    
    genus_dist = validate_genus_distribution(df)
    report_lines.extend([
        f"## Genus Distribution",
        f"### Exp1 (Sequence-level)",
        f"- Mean genera per fold: {genus_dist['fold_exp1']['mean_genera_per_fold']:.1f}",
        f"- Genera in all folds: {genus_dist['fold_exp1']['genera_in_all_folds']}",
        f"",
        f"### Exp2 (Species-level)",  
        f"- Mean genera per fold: {genus_dist['fold_exp2']['mean_genera_per_fold']:.1f}",
        f"- Genera in all folds: {genus_dist['fold_exp2']['genera_in_all_folds']}",
        f""
    ])
    
    report_path = output_path / f"validation_report_{dataset_name}.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Validation report saved to {report_path}")


def validate_dataset(output_dir: Path, dataset_name: str, dataset_type: str = 'full_10fold') -> None:
    """Validate a single dataset."""
    
    logger.info(f"Validating {dataset_name} ({dataset_type})")
    
    exp1_path = output_dir / 'exp1_sequence_fold' / dataset_type / f"{dataset_name}.csv"
    exp2_path = output_dir / 'exp2_species_fold' / dataset_type / f"{dataset_name}.csv"
    
    if not exp1_path.exists() or not exp2_path.exists():
        logger.warning(f"Dataset files not found for {dataset_name} ({dataset_type})")
        return
    
    df_exp1 = pd.read_csv(exp1_path)
    df_exp2 = pd.read_csv(exp2_path)
    
    if not df_exp1['sequence_id'].equals(df_exp2['sequence_id']):
        logger.error("Sequence IDs don't match between exp1 and exp2!")
        return
    
    df = df_exp1.copy()
    df['fold_exp2'] = df_exp2['fold_exp2']
    
    logger.info(f"Loaded {len(df):,} sequences")
    
    logger.info("\n" + "="*50)
    logger.info(f"VALIDATION: {dataset_name} ({dataset_type})")
    logger.info("="*50)
    
    validate_fold_balance(df, 'fold_exp1')
    validate_fold_balance(df, 'fold_exp2')
    validate_stratification(df, 'fold_exp1', 'species_resolution')
    validate_species_grouping(df)
    validate_genus_distribution(df)
    
    report_dir = output_dir / 'validation_reports'
    report_dir.mkdir(exist_ok=True)
    generate_summary_report(df, report_dir, f"{dataset_name}_{dataset_type}")


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
        logger.info("Running in LOCAL_TEST mode")
        output_dir = Path("../experiment_prep/output")
    else:
        experiments_dir = os.getenv('EXPERIMENTS_DIR')
        if not experiments_dir:
            raise ValueError("EXPERIMENTS_DIR must be set in .env")
        output_dir = Path(experiments_dir)
    
    # Actual file names from BLAST output
    datasets = [
        'b_recruited_99pct_species',
        'c_recruited_99pct_sp',
        'd_recruited_97pct_sp'
    ]
    
    for dataset_name in datasets:
        validate_dataset(output_dir, dataset_name, 'full')
        
        if config['debug_subset']['enabled']:
            validate_dataset(output_dir, dataset_name, 'debug_5genera')
    
    logger.info("\nValidation complete! Check validation_reports/ for detailed reports.")


if __name__ == "__main__":
    main()