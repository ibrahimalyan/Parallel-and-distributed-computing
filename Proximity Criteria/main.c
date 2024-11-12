// main.c

#include <stdio.h>   // Standard I/O library for basic input/output functions
#include <stdlib.h>  // Standard library for process control and memory allocation
#include <mpi.h>     // MPI library for parallel processing and message passing
#include <omp.h>     // OpenMP library for multi-threading within a process
#include "myProto.h" // Custom header file with function prototypes

// Main function initializes MPI, calls the main computation function, and finalizes MPI
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize the MPI environment, preparing it for use in parallel processing

    int world_size, world_rank;                 // Variables to hold the number of processes and the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the total number of processes involved in the MPI job
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank (ID) of the current process within the MPI environment

    // Check if the correct number of command line arguments are provided (input file and output file)
    if (argc != 3)
    {
        if (world_rank == 0)
        {                                                              // Only the master process (rank 0) prints the usage message to avoid duplicate outputs
            printf("Usage: %s <input_file> <output_file>\n", argv[0]); // Print the correct usage format
        }
        MPI_Finalize(); // Finalize the MPI environment to clean up before exiting
        return 1;       // Exit the program with a status indicating incorrect usage
    }

    // Master process prints the number of MPI processes and the number of OpenMP threads available
    if (world_rank == 0)
    {
        printf("Running on %d MPI processes with %d threads per process.\n", world_size, omp_get_max_threads());
        // This log message helps verify the parallel setup and the number of resources being utilized
    }

    // Call the function to check proximity criteria, passing the input and output file names
    checkProximityCriteria(argv[1], argv[2]);

    MPI_Finalize(); // Finalize the MPI environment, ensuring all resources are properly released
    return 0;       // Exit the program successfully
}
