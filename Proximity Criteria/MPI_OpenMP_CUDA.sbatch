#!/bin/bash
######################################################################################################################################
####################################### No Changes to this part of the batch #########################################################
######################################################################################################################################

#SBATCH --partition=rtx3070                 ### Specify partition name where to run a job.
#SBATCH --time=0-00:01:00                   ### Job running time limit. Make sure it is not exceeding the partition time limit! Format: D-H:MM:SS
#SBATCH --output=job-%J.out                 ### Output log for running job - %J is the job number variable
#SBATCH --error=job-%J.err                  ### Error log for running job - %J is the job number variable

##SBATCH --mail-user=user@post.jce.ac.il    ### User's email for sending job status
##SBATCH --mail-type=ALL                    ### Conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1                        ### Number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU
#SBATCH --cpus-per-task=8                   ### Number of CPU cores
#######################################################################################################################################


######################################################################################################################################
####################################### Start your code below ####  ###################################################################
######################################################################################################################################

#SBATCH --job-name='my_job'                 ### Name of the job. Replace 'my_job' with your desired job name
#SBATCH --ntasks=2                          ### Number of MPI ranks (tasks)

module load anaconda                        ### Load anaconda module (must present when working with conda environments)
source activate myenv                       ### Activating environment, environment must be configured before running the job

# Compilation steps
mpicxx -fopenmp -c main.c -o main.o
mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
/usr/local/cuda/bin/nvcc -I./Common -gencode arch=compute_86,code=sm_86 -c cudaFunctions.cu -o cudaFunctions.o
mpicxx -fopenmp -o mpiCudaOpenMP main.o cFunctions.o cudaFunctions.o -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart

# Running the program with MPI
mpirun -np 2 ./mpiCudaOpenMP input.txt output.txt
