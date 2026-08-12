#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name="TORL_cql"
#SBATCH --mail-user=lckjos003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=testlogs/testing_8.out
module load python/miniconda3-py3.12
unset PYTHONPATH
source activate torl_env
export WANDB_MODE=offline
pip freeze | grep -iE "torch|gymnasium|minari|wandb|tqdm|pyrallis|numpy"
python ExperimentA_HPC.py --algo cql --noise 0.25 --seed 11 --device cpu --steps 50000 --rew
