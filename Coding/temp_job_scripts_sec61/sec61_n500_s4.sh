#!/usr/bin/bash
module unload python
module load anaconda
cd /network/scratch/g/guangyuan.wang/fisher_random_walk/Coding
conda activate /network/scratch/g/guangyuan.wang/fisher_random_walk/condaenv
python exp_sec_61.py --n 500 --seed 4 --out_dir ./results_sec_61
