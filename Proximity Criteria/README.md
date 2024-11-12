# README.md

## MPI, OpenMP, and CUDA Integration

### Description

This project implements a parallel computing application that integrates MPI, OpenMP, and CUDA technologies.
The program is designed to perform distributed and parallel computations involving multiple processes, threads, and GPU acceleration.
The main objective is to efficiently handle large data sets using a combination of CPU and GPU resources.

### File Structure

1. **main.c** - The main source code file that coordinates the execution of MPI, OpenMP, and CUDA functions.
2. **cFunctions.c** - Contains auxiliary C functions called within the main program.
3. **cudaFunctions.cu** - CUDA source file with GPU-specific functions to accelerate calculations.
4. **helper_cuda.h** - Header file providing helper functions for CUDA operations.
5. **helper_string.h** - Header file for string handling functions used within the project.
6. **myProto.h** - Header file containing prototypes of functions used across different source files.
7. **input.txt** - Input file containing the parameters and data points used by the program. The file structure is as follows:
   ```
   N K D TCount
   id x1 x2 a b
   ...
   ```
   - `N`: Number of data points
   - `K`: Minimum number of points needed to satisfy the proximity criteria
   - `D`: Distance threshold
   - `TCount`: Number of time intervals for which calculations are performed
   - `id, x1, x2, a, b`: Details of each point including coordinates and additional parameters
8. **output.txt** - Output file where the results of the computations are saved.
9. **MPI_OpenMP_CUDA.sbatch** - Sample SBATCH script for submitting the job on a cluster environment.
10. **mpiCudaOpenMP** - Compiled executable of the program.
11. **job-8279.out** - Output log generated from the job submission.
12. **job-8279.err** - Error log generated from the job submission.

### Compilation and Running the Program

To compile and run the program, follow these steps:

1. **Compile the program**:

   ```bash
   mpicc main.c cFunctions.c cudaFunctions.cu -o mpiCudaOpenMP -fopenmp -lcuda -lcudart
   ```

   This command compiles the source files with MPI, OpenMP, and CUDA support.

2. **Run the program**:
   ```bash
   mpirun -np <number_of_processes> ./mpiCudaOpenMP
   ```
   Replace `<number_of_processes>` with the desired number of MPI processes.

### Sample Input (input.txt)

```
7 3 2.0 8
1 0.0 1.0 1.0 0.5
2 1.5 2.5 0.9 0.3
3 2.0 3.0 1.2 0.1
4 0.5 1.5 1.1 0.6
5 3.0 4.0 0.8 0.4
6 1.0 2.0 1.3 0.2
7 2.5 3.5 0.7 0.7
```

### Expected Output

The program outputs a list of points satisfying the proximity criteria for each time interval.
If no points meet the criteria, the program will indicate this in the output.

### Submitting the Job on a Cluster

Use the provided `MPI_OpenMP_CUDA.sbatch` script to submit the job on a cluster:

```bash
sbatch MPI_OpenMP_CUDA.sbatch
```

### Authors

- [Mousa Tams]
- [Ibrahim Alyan]
