#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=hml-lab1
#SBATCH --mail-type=END
#SBATCH --mail-user=cy1505@nyu.edu
#SBATCH --output=lab1.out
#SBATCH --partition=c01_17

echo "dp1:\n"
./dp1 1000000 1000
./dp1 300000000 20

echo "dp2:\n"
./dp2 1000000 1000
./dp2 300000000 20

echo "dp3:\n"
./dp3 1000000 1000
./dp3 300000000 20

module purge
module load numpy/python3.6/intel/1.14.0

echo "dp4:\n"
python3 dp4 1000000 1000
python3 dp4 300000000 20

echo "dp5:\n"
python3 dp5 1000000 1000
python3 dp5 300000000 20
