#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=hml-lab1
#SBATCH --mail-type=END
#SBATCH --mail-user=cy1505@nyu.edu
#SBATCH --output=sgd-no-bn.out
#SBATCH --gres=gpu:1

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5

./lab3
