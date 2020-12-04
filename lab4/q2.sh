#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=4:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=hml-lab1
#SBATCH --output=q2.out
#SBATCH --gres=gpu:k80:4
#SBATCH --reservation=chung

module purge
module load python3/intel/3.7.3
pip3 install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

echo "Q2:\n"

# echo "  batch size 32 with 1 gpu:\n"
# python3 lab4.py --epoch 2 --gpu --gpu_count 1 --batch_size 32 --workers 4 --disable_batch_norm
# echo "  batch size 128 with 1 gpu:\n"
# python3 lab4.py --epoch 2 --gpu --gpu_count 1 --batch_size 128 --workers 4 --disable_batch_norm
# echo "  batch size 512 with 1 gpu:\n"
# python3 lab4.py --epoch 2 --gpu --gpu_count 1 --batch_size 512 --workers 4 --disable_batch_norm
# echo "  batch size 2048 with 1 gpu:\n"
# python3 lab4.py --epoch 2 --gpu --gpu_count 1 --batch_size 2048 --workers 4 --disable_batch_norm

echo "  batch size 32 with 2 gpu:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 2 --batch_size 32 --workers 16 --disable_batch_norm
echo "  batch size 128 with 2 gpu:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 2 --batch_size 128 --workers 16 --disable_batch_norm
echo "  batch size 512 with 2 gpu:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 2 --batch_size 512 --workers 16 --disable_batch_norm

echo "  batch size 32 with 4 gpu:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 4 --batch_size 32 --workers 16 --disable_batch_norm
echo "  batch size 128 with 4 gpu:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 4 --batch_size 128 --workers 16 --disable_batch_norm
echo "  batch size 512 with 4 gpu:\n"
python3 lab4.py --epoch 2 --gpu --gpu_count 4 --batch_size 512 --workers 16 --disable_batch_norm


echo "##################\n"
echo "finish\n"
echo "##################\n"