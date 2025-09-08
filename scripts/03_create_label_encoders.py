#!/usr/bin/env python3
"""
03_create_label_encoders.py

Creates global label encoders from complete datasets for 
consistent model architectures
across all k-fold experiments. Reads configuration and paths 
from config.yaml and .env.
"""

import pandas as pd
import numpy as np
import yaml
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s -%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class LabelEncoder:
    """Simple label encoder for taxonomic levels"""

    def __init__(self, labels: List[str] = None):
        """Initialize encoder from list of labels"""
        self.label_to_index = {}
        self.index_to_label = {}

        if labels:
            unique_labels = sorted(set(labels))
            self.label_to_index = {label: idx for idx, label
in enumerate(unique_labels)}
            self.index_to_label = {idx: label for idx, label
in enumerate(unique_labels)}


def create_global_encoders():
    """
    Create global label encoders for all union types and 
dataset sizes
    """
    # Load environment and config
    load_dotenv("../.env")
    config = load_config()

    # Get paths
    data_base = Path(os.path.expandvars(os.getenv('FUNGAL_BASE'))) / 'experiments'

    # Union types to process
    union_types = ['standard', 'conservative']

    # Taxonomic levels
    taxonomic_levels = ['phylum', 'class', 'order', 'family',
'genus', 'species']

    for union_type in union_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING {union_type.upper()} UNION TYPE")
        logger.info(f"{'='*60}")

        # Process both full and debug datasets
        dataset_sizes = ['full', 'debug_5genera_10fold']

        for dataset_size in dataset_sizes:
            logger.info(f"\nProcessing {dataset_size} dataset...")

            # Build data path
            data_file = data_base / f"{union_type}.csv"
            if dataset_size == 'full':
                data_file = data_base / 'exp1_sequence_fold' /'full_10fold' / 'data' / f"{union_type}.csv"
            else:  # debug datasets  
                data_file = data_base / 'exp1_sequence_fold' /dataset_size / 'data' / f"{union_type}.csv"

            if not data_file.exists():
                logger.warning(f"Data file not found: {data_file}")
                continue

            # Load data
            logger.info(f"Loading data from {data_file}")
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} sequences")

            # Create encoders
            global_encoders = {}

            for level in taxonomic_levels:
                if level not in df.columns:
                    logger.warning(f"Column '{level}' not found in data")
                    continue

                # Get unique labels (excluding NaN)
                unique_labels =sorted(df[level].dropna().unique())
                logger.info(f"  {level}: {len(unique_labels)} unique labels")

                # Create encoder
                encoder = LabelEncoder(unique_labels)
                global_encoders[level] = {
                    'label_to_index': encoder.label_to_index,
                    'index_to_label': encoder.index_to_label
                }

            # Save encoders
            output_dir = data_file.parent
            output_file = output_dir / f"label_encoders_{union_type}_global.json"

            with open(output_file, 'w') as f:
                json.dump(global_encoders, f, indent=2)

            logger.info(f"Saved global encoders to {output_file}")

            # Print summary
            logger.info("Encoder summary:")
            for level, enc_data in global_encoders.items():
                logger.info(f"  {level}: {len(enc_data['label_to_index'])} classes")

def main():
    """Main execution"""
    logger.info("Creating global label encoders for k-fold experiments...")
    create_global_encoders()
    logger.info("Label encoder creation complete!")


if __name__ == "__main__":
    main()