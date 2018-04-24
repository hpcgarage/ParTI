#!/bin/bash -l

#SBATCH -J hicoo-mttkrp-mattile
#SBATCH -N 1         #Use 2 nodes
#SBATCH -t 00:10:00  #Set 30 minute time limit
#SBATCH -q debug   #Submit to the regular QOS
#SBATCH -L SCRATCH   #Job requires $SCRATCH file system
#SBATCH -C haswell   #Use Haswell nodes
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jiajiali@gatech.edu

# srun -n 32 -c 4 ./my_executable
# c = 64 / (n / N)
srun -n 1 -c 64 ../test_parti/run_parti_mttkrp_hicoo_matrixtiling.sh
