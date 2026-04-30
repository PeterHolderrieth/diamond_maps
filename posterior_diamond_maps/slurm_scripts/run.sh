#!/bin/bash

#SBATCH --job-name=diamond-maps
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1:00:00
#SBATCH --output=/data/user_data/%u/slurm_logs/diamond-maps/%j.out
#SBATCH --error=/data/user_data/%u/slurm_logs/diamond-maps/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5G

# Enable command echoing and exit on error
set -x
set -e

REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"

# Parse command line arguments
config_file="configs.cifar10.train"  # default
conda_env="diamond_maps"
launcher_file="learn.py"
PY_DIR="${REPO_ROOT}/py/launchers"
PY_ARGS=()
override_slurm_id="" # Added variable for the override

while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            PY_ARGS=("$@")
            break
            ;;
        --conda_env=*)
            conda_env="${1#*=}"
            shift
            ;;
        --py_dir=*)
            PY_DIR="${1#*=}"
            shift
            ;;
        --launcher_file=*)
            launcher_file="${1#*=}"
            shift
            ;;
        --slurm_id=*) # Added flag to capture override
            override_slurm_id="${1#*=}"
            shift
            ;;
        -*)
            echo "Unknown option $1" >&2
            exit 1
            ;;
        *)
            # Assume it's the config file if no other matches
            config_file="$1"
            shift
            ;;
    esac
done

# Set slurm params - handle both array and non-array jobs
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    # This is an array job
    if [[ -n "$override_slurm_id" ]]; then
        echo "ERROR: Cannot override slurm_id for an array job." >&2
        exit 1
    fi
    slurm_id=$SLURM_ARRAY_TASK_ID
    echo "Array job detected - Task ID: $slurm_id"
else
    # This is a regular job (use override if set, otherwise default to 0)
    slurm_id=${override_slurm_id:-0}
    echo "Regular job detected - Using task ID: $slurm_id"
fi
JOB_ID=${SLURM_JOB_ID:-manual}

# Detect actual number of GPUs allocated by SLURM (use this as source of truth)
if [[ -n "$SLURM_GPUS_ON_NODE" ]]; then
    num_gpus=$SLURM_GPUS_ON_NODE
elif [[ -n "$SLURM_GPUS" ]]; then
    num_gpus=$SLURM_GPUS
elif [[ -n "$SLURM_GPUS_PER_NODE" ]]; then
    num_gpus=$SLURM_GPUS_PER_NODE
else
    # Fallback: count CUDA devices if SLURM variables aren't set
    if command -v nvidia-smi &> /dev/null; then
        num_gpus=$(nvidia-smi --list-gpus | wc -l)
        echo "WARNING: SLURM GPU variables not set, detected $num_gpus GPUs via nvidia-smi"
    else
        num_gpus=1
        echo "WARNING: Could not detect GPUs, defaulting to 1"
    fi
fi

echo "Allocated GPUs (from SLURM): $num_gpus"

# Set up logging
LOG_BASE_DIR="/data/user_data/$USER/slurm_logs/diamond-maps"
mkdir -p "$LOG_BASE_DIR"

# Log partition type
echo "Job partition: $SLURM_JOB_PARTITION"

LOG_DIR="$LOG_BASE_DIR/$JOB_ID"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/log.txt"

# Redirect output to log file
exec >> "$LOG_FILE" 2>&1

echo "[$(date)] Job started on $(hostname)"
echo "Number of GPUs: $num_gpus"
echo "Config file: $config_file"

# Load modules
source /usr/share/Modules/init/bash
module purge
module load cuda-12.9

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$conda_env"
echo "Activated conda env: $(which python)"

# Change to working directory
cd "$PY_DIR" || { echo "Directory not found: $PY_DIR"; exit 1; }

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Build command arguments
CMD_ARGS=(
    "--cfg_path=$config_file"
    "--output_folder=$LOG_DIR"
    "--slurm_id=$slurm_id"
)

echo "Running with $num_gpus GPU(s)"
echo "Running command: python -u $launcher_file ${CMD_ARGS[*]} ${PY_ARGS[*]}"

# Run the training script directly (no srun)
python -u "$launcher_file" "${CMD_ARGS[@]}" "${PY_ARGS[@]}"

echo "[$(date)] Job completed."
