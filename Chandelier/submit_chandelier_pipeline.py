import subprocess
import itertools
import os

# ==========================================
# Directory Configuration
# ==========================================
OUT_DIR = "./results_chandelier"
JOB_SCRIPT_DIR = "./temp_job_scripts_chandelier"
SLURM_LOG_DIR = "./slurm_logs_chandelier"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(JOB_SCRIPT_DIR, exist_ok=True)
os.makedirs(SLURM_LOG_DIR, exist_ok=True)

PROJECT_DIR = "/network/scratch/g/guangyuan.wang/fisher_random_walk/Chandelier"
CONDA_ENV_PATH = "/network/scratch/g/guangyuan.wang/fisher_random_walk/condaenv" 

# ==========================================
# Parameter Grid
# ==========================================
config = {
    "samples": [1000],          # Full dataset scale
    "k_candidates": [20, 50]    # Testing m=20 and m=50 continuous treatments
}

# Increased time allocation for generating 1000*50 LLM generations and solving 1000 Riesz representers
slurm_time = "12:00:00" 

def main():
    keys, values = zip(*config.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Generating and submitting {len(experiments)} Chandelier Pipeline jobs...")
    
    for run_idx, exp in enumerate(experiments, start=1):
        s = exp["samples"]
        k = exp["k_candidates"]
        
        # Create an isolated output sub-directory for this specific grid configuration
        exp_out_dir = os.path.join(OUT_DIR, f"samples{s}_k{k}")
        os.makedirs(exp_out_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_out_dir, "plots"), exist_ok=True)
        
        # ---------------------------------------------------------
        # Define the sequential commands for Steps 1 through 6
        # ---------------------------------------------------------
        cmd_step1 = f"python step1_data_prep.py --samples {s} --k_candidates {k} --out_dir {exp_out_dir}"
        
        cmd_step2 = f"python step2_reward_extraction.py --in_file {exp_out_dir}/step1_treatments.json --out_file {exp_out_dir}/step2_rewards.json"
        
        cmd_step3 = f"python step3_green_density.py --in_file {exp_out_dir}/step2_rewards.json --out_dir {exp_out_dir} --epochs 300"
        
        cmd_step4_5 = f"python step4_5_testing_scaled.py --in_file {exp_out_dir}/step2_rewards.json --weights_file {exp_out_dir}/g_net_weights.pt --out_csv {exp_out_dir}/testing_results.csv"
        
        cmd_step6 = f"python step6_plotting.py --results_csv {exp_out_dir}/testing_results.csv --out_dir {exp_out_dir}/plots"
        
        # ---------------------------------------------------------
        # Build the SLURM bash script
        # ---------------------------------------------------------
        job_script_content = f"""#!/usr/bin/bash
set -e  # Abort the script immediately if any command fails

module unload python
module load anaconda
cd {PROJECT_DIR}
conda activate {CONDA_ENV_PATH}

# Force HuggingFace to use the network scratch space for heavy downloads
export HF_HOME="/network/scratch/g/guangyuan.wang/huggingface_cache"
export HF_DATASETS_CACHE="/network/scratch/g/guangyuan.wang/huggingface_cache/datasets"

echo "====================================="
echo "Starting Step 1: Data Prep & Sampling"
echo "====================================="
{cmd_step1}

echo "====================================="
echo "Starting Step 2: Reward Extraction"
echo "====================================="
{cmd_step2}

echo "====================================="
echo "Starting Step 3: Green Density Training"
echo "====================================="
{cmd_step3}

echo "====================================="
echo "Starting Steps 4 & 5: Scaled Testing"
echo "====================================="
{cmd_step4_5}

echo "====================================="
echo "Starting Step 6: Plotting"
echo "====================================="
{cmd_step6}

echo "Pipeline fully completed for samples={s}, k={k}!"
"""
        job_name = f"chandelier_s{s}_k{k}"
        job_script_filename = os.path.join(JOB_SCRIPT_DIR, f"{job_name}.sh")
        
        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)
            
        launch_command = (
            f"sbatch --job-name={job_name} --time={slurm_time} "
            f"--gres=gpu:rtx8000:1 -c 4 --mem=100G "
            f"--output={SLURM_LOG_DIR}/{job_name}.out {job_script_filename}"
        )
        
        subprocess.run(launch_command, shell=True, executable='/usr/bin/bash')

if __name__ == "__main__":
    main()