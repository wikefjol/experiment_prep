#!/usr/bin/env python3
"""
Orchestrate the experiment preparation pipeline.

This script runs all preparation steps in sequence:
1. Assign folds to all datasets
2. Create debug subsets
3. Validate fold assignments

Usage:
    # With HPC modules loaded (recommended):
    source ~/setup_env
    python run_pipeline.py
    
    # Or with local venv:
    source venv/bin/activate
    python run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_script(script_name: str) -> bool:
    """
    Run a script and check for errors.
    
    Args:
        script_name: Name of the script to run
        
    Returns:
        True if successful, False otherwise
    """
    script_path = Path("scripts") / script_name
    
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    start_time = time.time()
    
    # Run the script
    result = subprocess.run(
        [sys.executable, str(script_path)], 
        capture_output=False, 
        text=True
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ ERROR: {script_name} failed with code {result.returncode}")
        return False
    
    print(f"✓ {script_name} completed in {elapsed:.1f} seconds")
    return True


def check_environment() -> None:
    """Check that required packages are available."""
    required = ['pandas', 'numpy', 'sklearn', 'yaml']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Warning: Missing packages: {', '.join(missing)}")
        print("Please ensure your environment is properly set up:")
        print("  source ~/setup_env  # For HPC modules")
        print("  OR")
        print("  source venv/bin/activate && pip install -r requirements.txt")
        sys.exit(1)


def check_data_exists() -> bool:
    """Check if input data exists on mimer."""
    base_path = Path("/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full")
    
    required_files = [
        "a_recruited_99pct_90cov_species.csv",
        "b_recruited_99pct_90cov_sp_conservative.csv",
        "d_recruited_97pct_80cov_sp_permissive.csv"
    ]
    
    print("\n📁 Checking input data on mimer...")
    
    all_exist = True
    for filename in required_files:
        filepath = base_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {filename} NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Run all preparation steps in sequence."""
    print("\n" + "="*60)
    print("🧬 FUNGAL CLASSIFICATION EXPERIMENT PREPARATION")
    print("="*60)
    
    # Check environment
    print("\n🔍 Checking environment...")
    check_environment()
    print("  ✓ All required packages available")
    
    # Check data exists
    if not check_data_exists():
        print("\n❌ Missing input data files!")
        print("Please ensure BLAST filtered CSVs are in:")
        print("/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/")
        sys.exit(1)
    
    # Define pipeline scripts
    scripts = [
        ("01_assign_folds.py", "Assigning k-fold splits to all datasets"),
        ("02_create_debug_subset.py", "Creating 5-genera debug subsets"),
        ("03_validate_folds.py", "Validating fold assignments and generating reports")
    ]
    
    print("\n📋 Pipeline steps:")
    for i, (script, description) in enumerate(scripts, 1):
        print(f"  {i}. {description}")
    
    print("\n" + "="*60)
    print("🚀 Starting pipeline execution...")
    print("="*60)
    
    start_time = time.time()
    
    # Run each script
    for script, description in scripts:
        if not run_script(script):
            print(f"\n❌ Pipeline stopped due to error in {script}")
            sys.exit(1)
    
    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    # Success message
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total time: {minutes}m {seconds}s")
    print("="*60)
    
    print("\n📊 Output location:")
    print("/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/")
    print("\nDatasets are ready for training experiments!")


if __name__ == "__main__":
    main()