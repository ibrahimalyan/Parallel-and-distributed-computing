// cFunctions.c

#include "myProto.h" // Custom header file that contains function prototypes used in this file
#include <stdio.h>   // Standard Input/Output library for handling file operations
#include <stdlib.h>  // Standard library for memory allocation, process control, conversions, etc.
#include <math.h>    // Math library for mathematical functions such as sin()
#include <omp.h>     // OpenMP library for parallel programming within a process
#include <mpi.h>     // MPI library for distributed computing between multiple processes

// Declaration of the GPU distance calculation function defined elsewhere in cudaFunctions.cu
void gpuCalculateDistances(float *x, float *y, float *distances, int N);

// Main function that checks if points satisfy the proximity criteria based on the input file and writes the results to the output file
void checkProximityCriteria(char *inputFile, char *outputFile)
{
    // Open the input file in read mode to fetch parameters and point data
    FILE *input = fopen(inputFile, "r");
    if (input == NULL)
    {                                       // Check if the input file was opened successfully
        perror("Error opening input file"); // Print an error message if the file could not be opened
        exit(EXIT_FAILURE);                 // Exit the program with a failure status
    }

    // Initialize MPI to allow communication between different processes
    int world_rank, world_size;                 // Variables to store the rank of the current process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank (ID) of the current MPI process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the total number of MPI processes involved in the computation

    // Clear the contents of the output file by opening it in write mode
    FILE *output = fopen(outputFile, "w");
    if (output == NULL)
    {                                        // Check if the output file was opened successfully
        perror("Error opening output file"); // Print an error message if the file could not be opened
        exit(EXIT_FAILURE);                  // Exit the program with a failure status
    }
    fclose(output); // Close the output file after clearing its contents to ensure it starts fresh

    int N, K, TCount; // Variables to store the number of points, the required number of points to satisfy the criteria, and the number of t values to evaluate
    float D;          // Variable to store the maximum allowed distance for proximity
    // Read the first line of the input file to get the values of N, K, D, and TCount
    fscanf(input, "%d %d %f %d", &N, &K, &D, &TCount);
    printf("MPI Process %d: Read N=%d, K=%d, D=%f, TCount=%d\n", world_rank, N, K, D, TCount); // Log the read values for verification

    // Allocate memory dynamically for arrays to store the parameters of each point
    float *x1 = (float *)malloc(N * sizeof(float)); // Array for x1 values of points
    float *x2 = (float *)malloc(N * sizeof(float)); // Array for x2 values of points
    float *a = (float *)malloc(N * sizeof(float));  // Array for 'a' coefficients used in coordinate calculations
    float *b = (float *)malloc(N * sizeof(float));  // Array for 'b' coefficients used in coordinate calculations
    int *ids = (int *)malloc(N * sizeof(int));      // Array for point IDs

    // Loop to read each point's parameters from the input file and store them in the corresponding arrays
    for (int i = 0; i < N; i++)
    {
        fscanf(input, "%d %f %f %f %f", &ids[i], &x1[i], &x2[i], &a[i], &b[i]);                                            // Read the point ID, x1, x2, a, and b
        printf("MPI Process %d: Read point %d: x1=%f, x2=%f, a=%f, b=%f\n", world_rank, ids[i], x1[i], x2[i], a[i], b[i]); // Log the read point data
    }
    fclose(input); // Close the input file after reading all the required data

    // Allocate memory for arrays to store the calculated coordinates (x, y) and the pairwise distances between points
    float *x = (float *)malloc(N * sizeof(float));             // Array for calculated x coordinates of points
    float *y = (float *)malloc(N * sizeof(float));             // Array for calculated y coordinates of points
    float *distances = (float *)malloc(N * N * sizeof(float)); // 2D array (flattened) for storing distances between every pair of points

    // Log the size of the world (number of MPI processes) to ensure proper distribution of work
    printf("MPI Process %d: World size is %d\n", world_rank, world_size);

    // Determine the chunk of work (number of points) each MPI process will handle
    int chunk_size = N / world_size;                                   // Calculate the number of points per process
    int start = world_rank * chunk_size;                               // Calculate the starting index of the points for this process
    int end = (world_rank == world_size - 1) ? N : start + chunk_size; // Calculate the ending index, handling the last process specially to include any remaining points

    int foundAny = 0; // Flag to check if any points satisfy the criteria across all t values and processes

    // Iterate over each value of t to perform the proximity criteria checks
    for (int tIndex = 0; tIndex <= TCount; tIndex++)
    {
        double t = (2.0 * tIndex / ((float)(TCount))) - 1;                   // Calculate the value of t using the specified formula, ranging from -1 to 1
        printf("MPI Process %d: Calculating for t = %.6f\n", world_rank, t); // Log the value of t being processed

// Use OpenMP to parallelize the loop for calculating x and y coordinates for all points
#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            // Calculate the x coordinate using the given formula with x1, x2, and t
            x[i] = ((x2[i] - x1[i]) / 2.0) * sin(t * M_PI / 2.0) + (x2[i] + x1[i]) / 2.0;
            // Calculate the y coordinate using the linear equation with coefficients a and b
            y[i] = a[i] * x[i] + b[i];
        }

        // Use the GPU to calculate the distances between all pairs of points
        printf("MPI Process %d: Calling GPU for distance calculation\n", world_rank); // Log that the GPU function is being called
        gpuCalculateDistances(x, y, distances, N);                                    // Call the CUDA function to compute distances on the GPU
        printf("MPI Process %d: Finished GPU distance calculations\n", world_rank);   // Log that the GPU distance calculations are complete

        // Allocate memory to store indices of points that satisfy the proximity criteria for the current process
        int *localSatisfyingPoints = (int *)malloc(K * sizeof(int)); // Array to hold IDs of points that meet the criteria
        int localSatisfyingCount = 0;                                // Counter to keep track of how many points satisfy the criteria
        int stopEvaluation = 0;                                      // Flag to stop further evaluation once enough points have been found

        // Check proximity criteria for each point in the range assigned to the current MPI process
        for (int i = start; i < end && !stopEvaluation; i++)
        {
            int nearbyCount = 0; // Counter for points that are within the specified distance from the current point
            for (int j = 0; j < N; j++)
            {
                if (i != j && distances[i * N + j] < D)
                {                  // Check if the distance is less than the threshold D and not comparing the point with itself
                    nearbyCount++; // Increment the count of nearby points
                }
            }
            if (nearbyCount >= K)
            {                                                                                          // If the number of nearby points meets or exceeds K, the criteria are satisfied
                localSatisfyingPoints[localSatisfyingCount] = ids[i];                                  // Store the ID of the satisfying point
                localSatisfyingCount++;                                                                // Increment the count of satisfying points found locally
                printf("MPI Process %d: Point %d satisfies Proximity Criteria\n", world_rank, ids[i]); // Log that this point meets the criteria
            }
            if (localSatisfyingCount == K)
            { // Stop further checks once K satisfying points are found
                stopEvaluation = 1;
            }
        }

        // Allocate arrays to gather results from all MPI processes at the master process (rank 0)
        int *globalSatisfyingPoints = (int *)malloc(K * world_size * sizeof(int)); // Array to gather satisfying point IDs from all processes
        int *globalCounts = (int *)malloc(world_size * sizeof(int));               // Array to gather counts of satisfying points found by each process
        int found = (localSatisfyingCount == K) ? 1 : 0;                           // Flag indicating whether this process found enough points

        // Gather the flags indicating whether each process found sufficient satisfying points
        MPI_Gather(&found, 1, MPI_INT, globalCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // Gather the satisfying point IDs from each process to the master process
        MPI_Gather(localSatisfyingPoints, K, MPI_INT, globalSatisfyingPoints, K, MPI_INT, 0, MPI_COMM_WORLD);

        // Only the master process writes the results to the output file
        if (world_rank == 0)
        {
            for (int i = 0; i < world_size; i++)
            {
                if (globalCounts[i] == 1)
                {                                                                                     // Check if the current process found satisfying points
                    printf("MPI Process 0: Found %d points satisfying criteria at t = %.6f\n", K, t); // Log the satisfying criteria found
                    output = fopen(outputFile, "a");                                                  // Open the output file in append mode to add results
                    fprintf(output, "Points ");
                    for (int j = 0; j < K; j++)
                    {
                        fprintf(output, "%d", globalSatisfyingPoints[i * K + j]); // Write the IDs of satisfying points
                        if (j < K - 1)
                        {
                            fprintf(output, ", "); // Add a comma between IDs for formatting
                        }
                    }
                    fprintf(output, " satisfy Proximity Criteria at t = %.6f\n", t); // Write the t value associated with these points
                    fclose(output);                                                  // Close the output file after writing
                    foundAny = 1;                                                    // Set the flag indicating that satisfying points were found
                    break;                                                           // Stop writing once the first satisfying set is found for this t value
                }
            }
        }

        // Free memory allocated for this iteration to avoid memory leaks
        free(localSatisfyingPoints);
        free(globalSatisfyingPoints);
        free(globalCounts);
    }

    // If no points satisfy the criteria for any t value, the master process writes a message to the output file
    if (!foundAny && world_rank == 0)
    {
        output = fopen(outputFile, "a");
        fprintf(output, "There were no %d points found for any t.\n", K); // Inform that no points met the criteria
        fclose(output);
    }

    // Free all dynamically allocated memory to avoid memory leaks
    free(x1);
    free(x2);
    free(a);
    free(b);
    free(ids);
    free(x);
    free(y);
    free(distances);

    printf("MPI Process %d: Finished execution\n", world_rank); // Log that the current process has completed its task
}
