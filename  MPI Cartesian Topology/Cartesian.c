#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_FILENAME_LENGTH 100

// Structure to represent a point with x and y coordinates
struct Point
{
    double x;
    double y;
};

// Function prototypes
void read_input_file(const char *filename, double *D, int *MaxIterations, Point **points, int *num_points, int size);
void distribute_points(Point *points, int num_points, int rank, int size);
void gather_results(Point *points, int num_points, int rank, int size, MPI_Comm comm);
void calculate_new_point(Point *current, Point *left, Point *right, Point *up, Point *down, int rank, int size, MPI_Comm comm);
double calculate_distance(Point p);
void print_results(Point *points, int num_points, double D, int iterations, int MaxIterations);

int main(int argc, char *argv[])
{
    int rank, size;
    double D;
    int MaxIterations;
    Point *points = NULL;
    int num_points;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Process 0 reads the input file
    if (rank == 0)
    {
        read_input_file("data.txt", &D, &MaxIterations, &points, &num_points, size);
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MaxIterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for points array in non-root processes
    if (rank != 0)
    {
        points = (Point *)malloc(num_points * sizeof(Point));
    }

    // Broadcast points array to all processes
    MPI_Bcast(points, num_points * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

    int iterations = 0;
    int should_continue = 1;

    // Main iteration loop
    while (iterations < MaxIterations)
    {
        Point current = points[rank];
        Point left = current, right = current, up = current, down = current;

        // Calculate new point based on neighbors
        calculate_new_point(&current, &left, &right, &up, &down, rank, size, MPI_COMM_WORLD);

        points[rank] = current;
        double distance = calculate_distance(current);

        double max_distance;
        // Reduce to find the maximum distance among all processes
        MPI_Allreduce(&distance, &max_distance, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Print iteration and max distance for debugging (if needed)
        if (rank == 0)
        {
            // printf("Iteration %d, Max distance: %f\n", iterations, max_distance);
        }

        // Check stopping condition
        if (max_distance < D)
        {
            should_continue = 0;
        }

        // Broadcast the continuation flag to all processes
        MPI_Bcast(&should_continue, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!should_continue)
        {
            if (rank == 0)
            {
                // printf("Stopping condition met at iteration %d\n", iterations);
            }
            break;
        }

        iterations++;
    }

    // Gather results from all processes
    gather_results(points, num_points, rank, size, MPI_COMM_WORLD);

    // Print the final results
    if (rank == 0)
    {
        print_results(points, num_points, D, iterations, MaxIterations);
    }

    free(points);
    MPI_Finalize();
    return 0;
}

// Function to read input file
void read_input_file(const char *filename, double *D, int *MaxIterations, Point **points, int *num_points, int size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read D and MaxIterations from the first line
    fscanf(file, "%lf %d", D, MaxIterations);
    *num_points = size;
    *points = (Point *)malloc((*num_points) * sizeof(Point));

    // Read points from the file
    for (int i = 0; i < *num_points; i++)
    {
        fscanf(file, "%lf %lf", &((*points)[i].x), &((*points)[i].y));
    }

    fclose(file);
}

// Function to distribute points to all processes
void distribute_points(Point *points, int num_points, int rank, int size)
{
    MPI_Bcast(points, num_points * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
}

// Function to gather results from all processes
void gather_results(Point *points, int num_points, int rank, int size, MPI_Comm comm)
{
    MPI_Datatype MPI_Point;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_Point);
    MPI_Type_commit(&MPI_Point);

    if (rank != 0)
    {
        MPI_Sendrecv(&points[rank], 1, MPI_Point, 0, 0, &points[rank], 1, MPI_Point, 0, 0, comm, MPI_STATUS_IGNORE);
    }
    else
    {
        for (int i = 1; i < size; i++)
        {
            MPI_Sendrecv(&points[i], 1, MPI_Point, i, 0, &points[i], 1, MPI_Point, i, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    MPI_Type_free(&MPI_Point);
}

// Function to calculate the new point based on neighbors
void calculate_new_point(Point *current, Point *left, Point *right, Point *up, Point *down, int rank, int size, MPI_Comm comm)
{
    int K = (int)sqrt(size);
    int left_rank = (rank % K == 0) ? -1 : rank - 1;
    int right_rank = (rank % K == K - 1) ? -1 : rank + 1;
    int up_rank = (rank < K) ? -1 : rank - K;
    int down_rank = (rank >= size - K) ? -1 : rank + K;

    MPI_Datatype MPI_Point;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_Point);
    MPI_Type_commit(&MPI_Point);

    // Initialize neighbors to current point
    *left = *current;
    *right = *current;
    *up = *current;
    *down = *current;

    // Communicate with left neighbor if exists
    if (left_rank >= 0)
    {
        MPI_Sendrecv(current, 1, MPI_Point, left_rank, 0, left, 1, MPI_Point, left_rank, 0, comm, MPI_STATUS_IGNORE);
    }

    // Communicate with right neighbor if exists
    if (right_rank >= 0)
    {
        MPI_Sendrecv(current, 1, MPI_Point, right_rank, 0, right, 1, MPI_Point, right_rank, 0, comm, MPI_STATUS_IGNORE);
    }

    // Communicate with up neighbor if exists
    if (up_rank >= 0)
    {
        MPI_Sendrecv(current, 1, MPI_Point, up_rank, 0, up, 1, MPI_Point, up_rank, 0, comm, MPI_STATUS_IGNORE);
    }

    // Communicate with down neighbor if exists
    if (down_rank >= 0)
    {
        MPI_Sendrecv(current, 1, MPI_Point, down_rank, 0, down, 1, MPI_Point, down_rank, 0, comm, MPI_STATUS_IGNORE);
    }

    double total_x = 0.0, total_y = 0.0;
    int count = 0;

    // Sum the coordinates of all valid neighbors
    if (left->x != current->x || left->y != current->y)
    {
        total_x += left->x;
        total_y += left->y;
        count++;
    }
    if (right->x != current->x || right->y != current->y)
    {
        total_x += right->x;
        total_y += right->y;
        count++;
    }
    if (up->x != current->x || up->y != current->y)
    {
        total_x += up->x;
        total_y += up->y;
        count++;
    }
    if (down->x != current->x || down->y != current->y)
    {
        total_x += down->x;
        total_y += down->y;
        count++;
    }

    // Calculate the average
    if (count > 0)
    {
        current->x = total_x / count;
        current->y = total_y / count;
    }

    MPI_Type_free(&MPI_Point);
}

// Function to calculate the distance of a point from the origin
double calculate_distance(Point p)
{
    return sqrt(p.x * p.x + p.y * p.y);
}

// Function to print the results
void print_results(Point *points, int num_points, double D, int iterations, int MaxIterations)
{
    if (iterations >= MaxIterations)
    {
        printf("Stopped after maximum iterations\n");
    }
    else
    {
        printf("Where D=%.1lf , Max Iterations=%d\n\nFinal Results after %d iterations:\n", D, MaxIterations, iterations);

        for (int i = 0; i < num_points; i++)
        {
            double distance = calculate_distance(points[i]);
            printf("Point with rank %d is: (%.2f, %.2f),\tdistance=%.6f\n", i, points[i].x, points[i].y, distance);
        }
    }
}
