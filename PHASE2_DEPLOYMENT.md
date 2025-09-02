# Phase 2: HPC Deployment Guide

## Overview
This guide documents the deployment of the experiment_prep module to Alvis (cephyr) HPC cluster and processing the full BLAST output stored on mimer.

## Architecture Notes
- **Alvis (cephyr)**: Compute cluster where code runs (`/cephyr/users/filbern/Alvis`)
- **mimer**: Storage server for data (`/mimer/NOBACKUP/groups/snic2022-22-552/filbern`)
- **Important**: The scripts are designed to minimize I/O between systems - they read from mimer once, process in memory, then write back to mimer

## Prerequisites

### 1. Environment Setup on Alvis
```bash
# Go to workspace on Alvis
cd ~/workspace  # or use cwsp alias

# Copy the experiment_prep module from your local machine
# Option 1: git clone if you have it in a repo
git clone [your_repo]/experiment_prep.git
# Option 2: scp from your local machine
# scp -r local/path/to/experiment_prep filbern@alvis.c3se.chalmers.se:~/workspace/
cd experiment_prep

# Load Python module
module load Python/3.11.3-GCCcore-12.3.0  # or check available with: module spider Python

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify BLAST Output on mimer
```bash
# Check that BLAST outputs exist (from Alvis)
# Note: These paths assume your BLAST output is in v1.0/blast/filtered/full/
# Adjust if your actual BLAST results are elsewhere
ls -lh /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/
# Should see:
# - a_recruited_99pct_90cov_species.csv (~2.2M sequences)
# - b_recruited_99pct_90cov_sp_conservative.csv
# - d_recruited_97pct_80cov_sp_permissive.csv
```

## Deployment Steps

### Step 1: Update Configuration
Edit `config.yaml` in your workspace to point to mimer paths:
```yaml
input:
  # Adjust this path to match where your BLAST outputs actually are
  # Based on your tree output, you might have them in /mimer/.../filbern/data/blast_results/
  blast_output: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full"
  
output:
  base_path: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep"
```

### Step 2: Create SLURM Job Script
Create `run_experiment_prep.sh` in `~/workspace/experiment_prep/`:
```bash
#!/bin/bash
#SBATCH --job-name=exp_prep
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=exp_prep_%j.out
#SBATCH --error=exp_prep_%j.err

# Load modules
module load Python/3.11.3-GCCcore-12.3.0

# Navigate to workspace
cd /cephyr/users/filbern/Alvis/workspace/experiment_prep

# Activate virtual environment
source venv/bin/activate

echo "Starting experiment preparation at $(date)"
echo "Reading from mimer: /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full"
echo "Writing to mimer: /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep"

# Run the three scripts in sequence
echo "Step 1: Assigning folds..."
python scripts/01_assign_folds.py

if [ $? -eq 0 ]; then
    echo "Step 2: Creating debug subsets..."
    python scripts/02_create_debug_subset.py
    
    if [ $? -eq 0 ]; then
        echo "Step 3: Validating fold assignments..."
        python scripts/03_validate_folds.py
        echo "Experiment preparation completed successfully at $(date)"
    else
        echo "Error in debug subset creation"
        exit 1
    fi
else
    echo "Error in fold assignment"
    exit 1
fi
```

### Step 3: Submit Job from Alvis
```bash
# From ~/workspace/experiment_prep on Alvis
sbatch run_experiment_prep.sh
```

### Step 4: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor output
tail -f exp_prep_*.out

# Check for errors
tail -f exp_prep_*.err
```

## Expected Outputs

After successful completion, verify the following structure:

```
/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/
├── exp1_sequence_fold/
│   ├── full_10fold/
│   │   ├── a_recruited_99pct_90cov_species.csv
│   │   ├── b_recruited_99pct_90cov_sp_conservative.csv
│   │   └── d_recruited_97pct_80cov_sp_permissive.csv
│   └── debug_5genera_10fold/
│       ├── a_recruited_99pct_90cov_species.csv
│       ├── b_recruited_99pct_90cov_sp_conservative.csv
│       └── d_recruited_97pct_80cov_sp_permissive.csv
├── exp2_species_fold/
│   ├── full_10fold/
│   │   └── [same structure as exp1]
│   └── debug_5genera_10fold/
│       └── [same structure as exp1]
└── validation_reports/
    └── [validation reports for each dataset]
```

## Validation Checks

### 1. Check File Sizes
```bash
# Full datasets should be ~2.2M sequences (run from Alvis)
wc -l /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/exp1_sequence_fold/full_10fold/*.csv

# Debug subsets should be ~10k sequences
wc -l /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/exp1_sequence_fold/debug_5genera_10fold/*.csv
```

### 2. Verify Fold Columns
```python
import pandas as pd

# Quick check in Python (run from Alvis with venv activated)
df = pd.read_csv('/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/exp1_sequence_fold/full_10fold/a_recruited_99pct_90cov_species.csv')
print(f"Columns: {df.columns.tolist()}")
print(f"Fold distribution exp1:\n{df['fold_exp1'].value_counts().sort_index()}")
print(f"Fold distribution exp2:\n{df['fold_exp2'].value_counts().sort_index()}")
```

### 3. Check Species Grouping
```python
# Verify all sequences of a species are in same fold for exp2
species_folds = df.groupby('species')['fold_exp2'].nunique()
violations = species_folds[species_folds > 1]
print(f"Species with violations: {len(violations)}")
# Should be 0
```

## Troubleshooting

### Memory Issues
If job fails due to memory:
- Increase `--mem` in SLURM script (already set to 64G)
- Consider requesting more if processing very large datasets

### File Not Found
- Verify BLAST output paths exist
- Check permissions: `ls -la /path/to/blast/output`

### Pandas/NumPy Issues
- Ensure correct Python version (3.11.3 as loaded)
- Reinstall with: `pip install --upgrade pandas numpy scikit-learn pyyaml`

## Next Steps

Once Phase 2 is complete:
1. Verify all outputs are created correctly
2. Document the v1.0 dataset version in a README
3. Lock the directory to prevent accidental modifications:
   ```bash
   touch /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/LOCKED
   echo "Created on $(date)" >> /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep/LOCKED
   ```
4. Proceed to Phase 3: Training Refactor

## Contact

For issues or questions about the deployment:
- Check logs in `exp_prep_*.out` and `exp_prep_*.err`
- Verify all dependencies are installed
- Ensure sufficient compute resources are allocated