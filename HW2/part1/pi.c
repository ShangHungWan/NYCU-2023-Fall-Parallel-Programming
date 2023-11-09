#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <limits.h>
#include "mtwister.h"

int number_of_threads;

long long number_of_tosses;
long long number_in_circle;

pthread_mutex_t mutex;

double random_double(MTRand *r, double min, double max)
{
    return genRand(r) * (max - min) + min;
}

void *calculate_pi(void *num)
{
    MTRand r = seedRand(810 + (long long)num);

    long long start = (long long)(num) * (number_of_tosses / number_of_threads);
    long long end = (long long)(num + 1) * (number_of_tosses / number_of_threads);

    if ((long long)num == (long long)(number_of_threads - 1))
    {
        end = number_of_tosses;
    }

    long long private_number_in_circle = 0;
    for (long long i = start; i < end; i++)
    {
        double x = random_double(&r, -1.0, 1.0);
        double y = random_double(&r, -1.0, 1.0);
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
        {
            private_number_in_circle++;
        }
    }

    pthread_mutex_lock(&mutex);
    number_in_circle += private_number_in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: ./pi.out <number of threads> <number of tosses>\n");
        return 1;
    }

    char *endptr;
    long argv1 = strtol(argv[1], &endptr, 10);
    if (argv1 > INT_MAX || argv1 < INT_MIN)
    {
        printf("Number of threads could not out of int's bound\n");
        return 1;
    }
    if (*endptr != '\0')
    {
        printf("Number of threads is not a number\n");
        return 1;
    }
    number_of_threads = (int)argv1;

    long long argv2 = strtoll(argv[2], &endptr, 10);
    if (*endptr != '\0')
    {
        printf("Number of threads is not a number\n");
        return 1;
    }

    number_of_tosses = argv2;
    number_in_circle = 0;

    pthread_t *threads = malloc(sizeof(pthread_t) * number_of_threads);
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < number_of_threads; i++)
    {
        pthread_create(&threads[i], NULL, calculate_pi, (void *)(long long)i);
    }

    for (int i = 0; i < number_of_threads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    free(threads);

    printf("%f\n", 4 * number_in_circle / ((double)number_of_tosses));
}
