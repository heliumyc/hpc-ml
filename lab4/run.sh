#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=hml-lab1
#SBATCH --output=lab4.out
#SBATCH --gres=gpu:k80:4

module purge
module load python3/intel/3.7.3
pip3 install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

echo "  batch size 32:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 1 --batch_size 32 --workers 4 --disable_batch_norm
echo "  batch size 128:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 1 --batch_size 128 --workers 4 --disable_batch_norm

echo "##################\n"
echo "finish\n"
echo "##################\n"