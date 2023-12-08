#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    double offset = 2.0 / RAND_MAX;

    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    unsigned int x_seed = time(NULL) * (world_rank + 1);
    unsigned int y_seed = time(NULL) * (world_rank + 2);

    long long int start = (world_rank - 1) * tosses / world_size;
    long long int end = world_rank * tosses / world_size;
    long long int private_number_in_circle = 0, number_in_circle = 0;

    for (long long int i = start; i < end; i++)
    {
        double x = rand_r(&x_seed) * offset - 1.0;
        double y = rand_r(&y_seed) * offset - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
        {
            private_number_in_circle++;
        }
    }

    MPI_Win win;
    MPI_Win_create(&number_in_circle, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    MPI_Win_fence(0, win);
    MPI_Accumulate(&private_number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
    MPI_Win_fence(0, win);

    if (world_rank == 0)
    {
        pi_result = 4.0 * number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Win_free(&win);

    MPI_Finalize();
    return 0;
}