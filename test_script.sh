#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=test_output.txt
#SBATCH --error=test_error.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00

echo "Hello World"
