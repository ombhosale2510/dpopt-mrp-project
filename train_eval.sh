#!/bin/bash
#SBATCH --time=0-04:45:00
#SBATCH --account=def-wzhang25
#SBATCH --mem-per-cpu=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Record start time
start_time=$(date +%s)

echo "Hello world !"
nvidia-smi

# Load python and cuda modules
module load python/3.10 cuda cudnn

module load StdEnv/2023
module load cudacore/.12.2.2

module load pyarrow/14.0.0

# Activate env
source ~/envs/DP-OPT/bin/activate

logdir=/home/ombh/scratch/saved
#datadir=home/ombh/scratch/data

# Declare the .sh file name with a distinct echo statement
echo "========================================="
echo -e "\033[1;32mRUNNING: train_opt.sh\033[0m"
echo -e "\033[1;32mMODEL: vicuna-7b-v1.5\033[0m"
echo "========================================="

python train_opt.py --ape_mode=iid_ibwd \
        --ensemble_gen=True \
        --gen_temp=1.1 \
        --num_prompt=40 \
        --max_new_tokens=50 \
        --data=sst2 \
        --seed=1 \
        --holdout_ratio=0.01 \
        --no_wandb \
        --device=cuda \
        --model='./vicuna-7b-v1.5/'

python eval_opt.py \
        --ape_mode=iid_ibwd \
        --ensemble_gen=True \
        --gen_temp=1.1 \
        --seed=1 \
        --num_prompt=40 \
        --max_new_tokens=50 \
        --data=sst2 \
        --no_wandb \
        --device=cuda \
        --test_model='./vicuna-7b-v1.5/'

# Record end time
end_time=$(date +%s)

# Calculate and print elapsed time
elapsed=$(( end_time - start_time ))
echo "Elapsed time: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s"


echo "##################"
