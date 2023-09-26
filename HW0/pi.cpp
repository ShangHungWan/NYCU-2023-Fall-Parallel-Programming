#include <iostream>
#include <cstdlib>
#include <omp.h>

using namespace std;

double random_double(double min, double max)
{
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

int main()
{
    long long int number_of_tosses = 10000000;
    long long int number_in_circle = 0;

#pragma omp parallel for reduction(+ : number_in_circle)
    for (long long int i = 0; i < number_of_tosses; i++)
    {
        double x = random_double(-1.0, 1.0);
        double y = random_double(-1.0, 1.0);
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
        {
            number_in_circle++;
        }
    }

    cout << "Pi: " << 4 * number_in_circle / ((double)number_of_tosses) << endl;
}