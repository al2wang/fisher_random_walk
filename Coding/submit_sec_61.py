import subprocess
import itertools
import os

OUT_DIR = "./results_sec_61"
JOB_SCRIPT_DIR = "./temp_job_scripts_sec61"
SLURM_LOG_DIR = "./slurm_logs_sec61"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(JOB_SCRIPT_DIR, exist_ok=True)
os.makedirs(SLURM_LOG_DIR, exist_ok=True)

PROJECT_DIR = "/network/scratch/g/guangyuan.wang/fisher_random_walk/Coding"
CONDA_ENV_PATH = "/network/scratch/g/guangyuan.wang/fisher_random_walk/condaenv" 

# Parameter grid based on Section 6.1 of the writeup
config = {
    "sample_sizes": [500, 1000, 2000, 5000],
    "seeds": list(range(1, 11)) # 10 independent replications per sample size
}

slurm_time = "4:00:00"

def main():
    keys, values = zip(*config.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Generating and submitting {len(experiments)} Policy Evaluation jobs...")
    
    for run_idx, exp in enumerate(experiments, start=1):
        n, seed = exp["sample_sizes"], exp["seeds"]
        
        # Calling the script with argparse flags
        python_run_command = (
            f"python exp_sec_61.py "
            f"--n {n} --seed {seed} --out_dir {OUT_DIR}"
        )
        
        job_script_content = f"""#!/usr/bin/bash
module unload python
module load anaconda
cd {PROJECT_DIR}
conda activate {CONDA_ENV_PATH}
{python_run_command}
"""
        job_name = f"sec61_n{n}_s{seed}"
        job_script_filename = os.path.join(JOB_SCRIPT_DIR, f"{job_name}.sh")
        
        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)
            
        launch_command = (
            f"sbatch --job-name={job_name} --time={slurm_time} "
            f"--gres=gpu:1 -c 4 --mem=64G "
            f"--output={SLURM_LOG_DIR}/{job_name}.out {job_script_filename}"
        )
        subprocess.run(launch_command, shell=True, executable='/usr/bin/bash')

if __name__ == "__main__":
    main()