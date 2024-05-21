#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="GP_MASTER"
#SBATCH --get-user-env
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#                               # #SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=128    # 24 thin, 128 epyc	
#SBATCH --mem=450G
#SBATCH --time=36:00:00
#SBATCH --nodelist=epyc001

# REMARK: 
#    The master process does not need so much resources. 
#    All the computational cost will be distributed among 
#    slaves. It only initialize the GP object and coordinates
#    all the other processes.

# Standard preamble
echo "---------------------------------------------"
echo "SLURM job ID:        $SLURM_JOB_ID"
echo "SLURM job node list: $SLURM_JOB_NODELIST"
echo "DATE:                $(date)"
echo "HOSTNAME:            $(hostname)"
echo "---------------------------------------------"


# Load the venv 
source /u/dssc/ipasia00/test_dask/dask_epyc/bin/activate
# source /u/dssc/ipasia00/test_dask/dask/bin/activate

# export OMP_NUM_THREADS=16
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close

export DASK_WORKER_PROCESSES=128


# execute the code
python3 -u infer_2_4.py 

