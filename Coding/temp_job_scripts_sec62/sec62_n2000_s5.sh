#!/usr/bin/bash
module unload python
module load anaconda
cd /network/scratch/g/guangyuan.wang/fisher_random_walk/Coding
conda activate /network/scratch/g/guangyuan.wang/fisher_random_walk/condaenv
python exp_sec_62.py --n 2000 --seed 5 --out_dir ./results_sec_62
