#!/usr/bin/bash
set -e  # Abort the script immediately if any command fails

module unload python
module load anaconda
cd /network/scratch/g/guangyuan.wang/fisher_random_walk/Chandelier
conda activate /network/scratch/g/guangyuan.wang/fisher_random_walk/condaenv

# Force HuggingFace to use the network scratch space for heavy downloads
export HF_HOME="/network/scratch/g/guangyuan.wang/huggingface_cache"
export HF_DATASETS_CACHE="/network/scratch/g/guangyuan.wang/huggingface_cache/datasets"

echo "====================================="
echo "Starting Step 1: Data Prep & Sampling"
echo "====================================="
python step1_data_prep.py --samples 1000 --k_candidates 50 --out_dir ./results_chandelier/samples1000_k50

echo "====================================="
echo "Starting Step 2: Reward Extraction"
echo "====================================="
python step2_reward_extraction.py --in_file ./results_chandelier/samples1000_k50/step1_treatments.json --out_file ./results_chandelier/samples1000_k50/step2_rewards.json

echo "====================================="
echo "Starting Step 3: Green Density Training"
echo "====================================="
python step3_green_density.py --in_file ./results_chandelier/samples1000_k50/step2_rewards.json --out_dir ./results_chandelier/samples1000_k50 --epochs 300

echo "====================================="
echo "Starting Steps 4 & 5: Scaled Testing"
echo "====================================="
python step4_5_testing_scaled.py --in_file ./results_chandelier/samples1000_k50/step2_rewards.json --weights_file ./results_chandelier/samples1000_k50/g_net_weights.pt --out_csv ./results_chandelier/samples1000_k50/testing_results.csv

echo "====================================="
echo "Starting Step 6: Plotting"
echo "====================================="
python step6_plotting.py --results_csv ./results_chandelier/samples1000_k50/testing_results.csv --out_dir ./results_chandelier/samples1000_k50/plots

echo "Pipeline fully completed for samples=1000, k=50!"
