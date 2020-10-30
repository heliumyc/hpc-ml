#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=hml-lab1
#SBATCH --mail-type=END
#SBATCH --mail-user=cy1505@nyu.edu
#SBATCH --output=lab2.out
#SBATCH --partition=c39_01

module purge
module load python3/intel/3.7.3
pip3 install --user torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

echo "c2:\n"
python3 lab2.py --epoch 5

# echo "\n ======================== \n"
# echo "c3:\n"
# echo "workers 0 \n"
# python3 lab2.py --epoch 5 --workers 0
# echo "workers 4 \n"
# python3 lab2.py --epoch 5 --workers 4
# echo "workers 8 \n"
# python3 lab2.py --epoch 5 --workers 8
# echo "workers 12 \n"
# python3 lab2.py --epoch 5 --workers 12
# echo "workers 16 \n"
# python3 lab2.py --epoch 5 --workers 16
# echo "workers 20 \n"
# python3 lab2.py --epoch 5 --workers 20


# echo "\n ======================== \n"
# echo "c4:\n"
# echo "workers 1 \n"
# python3 lab2.py --epoch 5 --workers 1

# echo "\n ======================== \n"
# echo "c5:\n"
