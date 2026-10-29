#!/bin/bash
#SBATCH --job-name=cellranger_gex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150gb
#SBATCH --partition=24
#SBATCH --account=weissman
#SBATCH --output=cellranger_gex-%j.out
#SBATCH --array=1-2

set -euo pipefail

## Configure
CSV="samples.csv"
cellranger_path="/lab/solexa_weissman/reubens/software/cellranger-9.0.1/bin/cellranger"
transcriptome="/lab/solexa_weissman/wcolgan/reference/cellranger/mm39_PETS"
transcriptome=$(realpath "$transcriptome")
fastq_root="/lab/solexa_weissman/fame/260814_crispants_batch1/fastq"
fastq_root=$(realpath "$fastq_root")

# Read the Nth data row (skip header). Task 1 -> first data row.
line="$(awk -F',' -v row="$SLURM_ARRAY_TASK_ID" 'NR==row+1 {print; exit}' "$CSV")"
if [[ -z "${line}" ]]; then
  echo "No CSV row found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi
sample="$(printf '%s' "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')"

# Sample names are "<Perturbation>_T<gem#>" (e.g. Tbx5_T2); the matching fastqs
# live in fastq/<Perturbation>/GEM<gem#>/.
pert="${sample%_T*}"
gem_num="${sample##*_T}"
fastq_dir="${fastq_root}/${pert}/GEM${gem_num}"

# Library-specific cellranger run id, so pipestance dirs never collide.
run_id="${sample}_gex"

# Template for the libraries.csv content
libraries_template="fastqs,sample,library_type
{fastq_dir},{sample},Gene Expression"

mkdir -p ./data

# Run cellranger count
echo "Running cellranger count for sample: ${sample}"
libraries_csv="${run_id}_libraries.csv"
echo "$libraries_template" | sed "s/{sample}/$sample/g" | sed "s|{fastq_dir}|$fastq_dir|g" > "$libraries_csv"
$cellranger_path count \
  --id="$run_id" \
  --libraries="$libraries_csv" \
  --transcriptome="$transcriptome" \
  --create-bam=false \
  --localmem=150 \
  --localcores=16 \
  --nosecondary

output_dir="./$run_id/outs"
data_dir="./data/${sample}"
mkdir -p "$data_dir"

# Clean up
rm -f "$libraries_csv"
# Counts
filtered_h5="${output_dir}/filtered_feature_bc_matrix.h5"
mv -f "$filtered_h5" "${data_dir}/${sample}_filtered_counts.h5"
# Barcodes
barcodes="${output_dir}/filtered_feature_bc_matrix/barcodes.tsv.gz"
mv -f "$barcodes" "${data_dir}/${sample}_barcodes.tsv.gz"
# Summary
summary="${output_dir}/web_summary.html"
mv -f "$summary" "${data_dir}/${sample}_summary.html"
# Remove output directory
rm -r "./${run_id}"
