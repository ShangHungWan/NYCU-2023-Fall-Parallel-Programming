#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    double offset = (1.0 / RAND_MAX) * 2.0;

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
    long long int private_number_in_circle = 0;

    if (world_rank > 0)
    {
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

        MPI_Send(&private_number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
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

        for (int i = 1; i < world_size; i++)
        {
            long long int number_in_circle = 0;
            MPI_Recv(&number_in_circle, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            private_number_in_circle += number_in_circle;
        }

        pi_result = 4.0 * private_number_in_circle / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
