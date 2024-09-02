
# README.txt

## MPI Cartesian Topology and User Defined Types

### Description
This project implements an MPI-based program to practice MPI Cartesian topology and user-defined types. The main objective is to distribute points among MPI processes and iteratively update these points based on their neighbors until a specified distance criterion is met or the maximum number of iterations is reached.

### File Structure
1. **Cartesian.c** - The main source code file containing the implementation of the MPI program.
2. **data.txt** - Input file containing the initial parameters and points.
3. **mySbatch** - Sample sbatch script for submitting the job on a cluster.
4. **README.txt** - This file, containing instructions for compiling and running the program.

### Input File (data.txt)
The input file should be named `data.txt` and placed in the same directory as the executable. The file should have the following structure:
```
D    MaxIterations
x1   y1
x2   y2
x3   y3
...
xN2  yN2
```
Where:
- `D` is the distance threshold.
- `MaxIterations` is the maximum number of iterations.
- `x1 y1, x2 y2, ..., xN2 yN2` are the coordinates of the points.

### Compilation and run the program on the cluster use the following command
sbatch mySbatch Cartesian


```
Ensure that the number of processes (`<number_of_processes>`) is a perfect square (e.g., 4, 9, 16) to form a KxK Cartesian grid By Adjusting the Data.txt and The Sbatch files.

### Sample Input (data.txt) for 9 processes
```
150.4   5000
1.2    34.33
-3.3   5.5
34.54  -2
23.73  444.3
100.2  -10.5
12     100 
2.56   15
100    22
15.6   -1
```

### Expected Output
If The Program Reaches The Max Iterations It Prints "Stopped after maximum iterations".

If this distance is less than the given value D for all processes - the program ends and the process 0 displays all current points Pi received from each process according to their ranks – in order from the first to the last.


```
Where D=150.4 , Max Iterations=5000

Final Results after 6 iterations:
Point with rank 0 is: (32.26, 145.99),	distance=149.507050
Point with rank 1 is: (42.49, 4.33),	distance=42.712388
Point with rank 2 is: (32.04, 139.61),	distance=143.239175
Point with rank 3 is: (42.10, 4.54),	distance=42.341643
Point with rank 4 is: (33.11, 142.95),	distance=146.733803
Point with rank 5 is: (42.67, 3.90),	distance=42.847365
Point with rank 6 is: (34.17, 146.29),	distance=150.228982
Point with rank 7 is: (42.27, 4.11),	distance=42.473991
Point with rank 8 is: (33.96, 139.91),	distance=143.976118


```

### sbatch Script (mySbatch)
Here is an example of an sbatch script to submit the job on a cluster:
```#!/bin/bash

###################################################################################
####################  NO CHANGES FOR THIS PART OF THE BATCH #######################
###################################################################################

#SBATCH --partition main            ### Specify partition name where to run a job.
#SBATCH --time 0-00:05:00           ### Job running time limit. Make sure it is not exceeding the partition time limit! Format: D-H:MM:SS
#SBATCH --output job-%J.out         ### Output log for running job - %J is the job number variable
##SBATCH --mail-user=user@post.jce.ac.il  ### User's email for sending job status
##SBATCH --mail-type=ALL            ### Conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1                ### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU
#SBATCH --cpus-per-task=1           ### number of CPU cores

###################################################################################
####################  Start your code below #######################################
###################################################################################

#SBATCH --ntasks=9                  ### Total number of tasks/processes
#SBATCH --job-name 'Cartesian'      ### Name of the job. replace my_job with your desired job name
module load anaconda                ### load anaconda module (must present when working with conda environments)
source activate myenv               ### activating environment, environment must be configured before running the job

### Compile your program ###
mpicxx Cartesian.c -o Cartesian

### Run your program ###
mpirun ./Cartesian

```
Save this script as `mySbatch` and submit it using the command:
```sbatch mySbatch Cartesian
```

### Authors
- [Mousa Tams]
- [Ibrahim Alyan]

