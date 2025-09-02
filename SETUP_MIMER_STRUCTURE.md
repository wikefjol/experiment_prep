# Setting Up Mimer Directory Structure

## Current Situation
- Filtered CSV files exist in: `~/workspace/blast/blast_results/full/filtered_results_csv/`
- Need to organize them properly on mimer for the experiment_prep pipeline

## Step 1: Create Directory Structure on Mimer

```bash
# From Alvis, create the v1.0 structure on mimer
mkdir -p /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full

# Create directories for experiment outputs
mkdir -p /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/experiment_prep
```

## Step 2: Copy Filtered CSVs to Mimer

The experiment_prep scripts expect files with specific names. We need to rename them during copy:

```bash
# Navigate to your filtered results
cd ~/workspace/blast/blast_results/full/filtered_results_csv/

# Copy and rename the files to match expected names
cp b_recruited_99pct_species.csv /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/a_recruited_99pct_90cov_species.csv

cp c_recruited_99pct_sp.csv /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/b_recruited_99pct_90cov_sp_conservative.csv

cp d_recruited_97pct_sp.csv /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/d_recruited_97pct_80cov_sp_permissive.csv

# Also copy the master file for reference
cp all_sequences_master.csv /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/

# Copy statistics for documentation
cp recruitment_statistics.json /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/
cp recruitment_statistics_report.txt /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/
```

## Step 3: Verify the Files

```bash
# Check that files were copied correctly
ls -lh /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/

# Quick check of row counts
wc -l /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/*.csv

# Verify column headers (should include sequence_id, species, genus, etc.)
head -n 1 /mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full/a_recruited_99pct_90cov_species.csv
```

## Step 4: Update BLAST Config for Future Runs

To ensure future BLAST filtering saves directly to mimer, update your `~/workspace/blast/config.yaml`:

```yaml
# Add or update output paths section
output:
  # Local workspace for temporary/working files
  workspace: "/cephyr/users/filbern/Alvis/workspace/blast/blast_results"
  
  # Final storage on mimer
  mimer_base: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification"
  
  # Version-specific paths
  current_version: "v1.0"
  filtered_output: "/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/blast/filtered/full"
```

## Step 5: Create a Transfer Script for Future Use

Create `~/workspace/blast/scripts/transfer_to_mimer.sh`:

```bash
#!/bin/bash
# Script to transfer filtered results from workspace to mimer

VERSION=${1:-v1.0}
SOURCE_DIR="$HOME/workspace/blast/blast_results/full/filtered_results_csv"
TARGET_DIR="/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/$VERSION/blast/filtered/full"

echo "Transferring filtered BLAST results to mimer..."
echo "Version: $VERSION"
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Create target directory
mkdir -p "$TARGET_DIR"

# Copy with renaming
cp "$SOURCE_DIR/b_recruited_99pct_species.csv" "$TARGET_DIR/a_recruited_99pct_90cov_species.csv"
cp "$SOURCE_DIR/c_recruited_99pct_sp.csv" "$TARGET_DIR/b_recruited_99pct_90cov_sp_conservative.csv"
cp "$SOURCE_DIR/d_recruited_97pct_sp.csv" "$TARGET_DIR/d_recruited_97pct_80cov_sp_permissive.csv"

# Copy additional files
cp "$SOURCE_DIR/all_sequences_master.csv" "$TARGET_DIR/"
cp "$SOURCE_DIR/recruitment_statistics.json" "$TARGET_DIR/"
cp "$SOURCE_DIR/recruitment_statistics_report.txt" "$TARGET_DIR/"

echo "Transfer complete!"
ls -lh "$TARGET_DIR/"
```

Make it executable:
```bash
chmod +x ~/workspace/blast/scripts/transfer_to_mimer.sh
```

## Step 6: Document the Pipeline

Create `/mimer/NOBACKUP/groups/snic2022-22-552/filbern/fungal_classification/v1.0/README.md`:

```markdown
# Fungal Classification v1.0 Dataset

## Data Pipeline
1. **BLAST Search**: Run on Alvis, results stored temporarily in workspace
2. **Filtering**: Process BLAST results to create recruitment categories
3. **Transfer**: Move filtered CSVs from Alvis workspace to mimer
4. **Experiment Prep**: Add fold assignments and create train/test splits
5. **Training**: Use prepared datasets for model training

## Directory Structure
```
v1.0/
├── blast/
│   ├── raw/              # Original BLAST output (TSV)
│   └── filtered/         # Filtered and categorized sequences
│       └── full/
│           ├── a_recruited_99pct_90cov_species.csv
│           ├── b_recruited_99pct_90cov_sp_conservative.csv
│           └── d_recruited_97pct_80cov_sp_permissive.csv
└── experiment_prep/      # K-fold splits and debug subsets
    ├── exp1_sequence_fold/
    └── exp2_species_fold/
```

## File Naming Convention
- Original BLAST filter names → Experiment prep names
- `b_recruited_99pct_species.csv` → `a_recruited_99pct_90cov_species.csv`
- `c_recruited_99pct_sp.csv` → `b_recruited_99pct_90cov_sp_conservative.csv`
- `d_recruited_97pct_sp.csv` → `d_recruited_97pct_80cov_sp_permissive.csv`

Created: $(date)
```

## Next Steps
After completing these steps:
1. Run the experiment_prep pipeline using the deployment guide
2. The config.yaml in experiment_prep should point to these mimer paths
3. All future processing will read from and write to mimer