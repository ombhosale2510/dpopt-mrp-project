#SBATCH --time=01:30:00
#SBATCH --account=def-wzhang25
#SBATCH --mem-per-cpu=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10

# Load python and cuda modules
module load python/3.10 cuda cudann

# Activate env
source ~/envs/DP-OPT/bin/activate


python train_opt.py --ape_mode=iid_ibwd --ensemble_gen=True --gen_temp=1.1 --num_prompt=40 --max_new_tokens=50 --data=sst2 --holdout_ratio=0.01