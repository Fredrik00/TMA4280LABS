#!/bin/sh
#SBATCH --partition=TRAINING
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -c 2
#SBATCH --mem=12000
# Memory per node specification is in MB. It is optional.
# The default limit is 3000MB per core.
#SBATCH --job-name="poisson"
#SBATCH --output=poisson-srun.out
#SBATCH --mail-user=fredrva@stud.ntnu.no
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load GCC OpenMPI
mpirun ./poisson cvtest
