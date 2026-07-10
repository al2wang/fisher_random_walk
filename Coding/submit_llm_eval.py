import subprocess
import itertools
import os

OUT_DIR = "./results_llm_eval"
JOB_SCRIPT_DIR = "./temp_job_scripts_llm"
SLURM_LOG_DIR = "./slurm_logs_llm"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(JOB_SCRIPT_DIR, exist_ok=True)
os.makedirs(SLURM_LOG_DIR, exist_ok=True)

PROJECT_DIR = "/network/scratch/g/guangyuan.wang/fisher_random_walk/Coding"
CONDA_ENV_PATH = "/network/scratch/g/guangyuan.wang/fisher_random_walk/condaenv" 

# Parameter grid: Testing different training epoch durations
config = {
    "epochs": [1500, 3000, 4500, 6000] # Adjust these test values as needed
}

slurm_time = "6:00:00"

def main():
    keys, values = zip(*config.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Generating and submitting {len(experiments)} LLM Evaluation jobs...")
    
    for run_idx, exp in enumerate(experiments, start=1):
        e = exp["epochs"]
        
        # Create a specific output sub-directory for this epoch setting 
        # to prevent concurrent jobs from overwriting the same CSVs
        epoch_out_dir = os.path.join(OUT_DIR, f"ep_{e}")
        os.makedirs(epoch_out_dir, exist_ok=True)
        
        # Calling the LLM eval script with the dynamic epoch and out_dir flags
        python_run_command = (
            f"python exp_llm_eval.py "
            f"--epochs {e} --out_dir {epoch_out_dir}"
        )
        
        job_script_content = f"""#!/usr/bin/bash
module unload python
module load anaconda
cd {PROJECT_DIR}
conda activate {CONDA_ENV_PATH}
{python_run_command}
"""
        job_name = f"llm_eval_ep{e}"
        job_script_filename = os.path.join(JOB_SCRIPT_DIR, f"{job_name}.sh")
        
        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)
            
        launch_command = (
            f"sbatch --job-name={job_name} --time={slurm_time} "
            f"--gres=gpu:rtx8000:1 -c 4 --mem=64G "
            f"--output={SLURM_LOG_DIR}/{job_name}.out {job_script_filename}"
        )
        subprocess.run(launch_command, shell=True, executable='/usr/bin/bash')

if __name__ == "__main__":
    main()