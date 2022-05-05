#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <omp.h>

#define ACCURACY 0.01
#define NUM_THREADS 8
#define NUM_TASKS 200

void Usage(char *prog_name);

double pi_calc(long long, long long, double, double);
double pi_calc_parallel(long long, long long, double, double);

int main(int argc, char *argv[])
{
    long long n, i;
    double factor;
    double sum = 0.0;
    double par_time1, par_elapsed;
    double seq_time1, seq_elapsed;
    double pi_par, pi_seq;

    if (argc != 2)
        Usage(argv[0]);
    n = strtoll(argv[1], NULL, 10);
    if (n < 1)
        Usage(argv[0]);

    printf("With n = %lld terms and %d threads in parallel code: \n", n, NUM_THREADS);
    seq_time1 = omp_get_wtime();
    pi_seq = pi_calc(n, i, factor, sum);
    seq_elapsed = omp_get_wtime() - seq_time1;

    par_time1 = omp_get_wtime();
    pi_par = pi_calc_parallel(n, i, factor, sum);
    par_elapsed = omp_get_wtime() - par_time1;

    printf("Sequential estimate of pi = %.14f\n", pi_seq);
    printf("Parallel estimate of pi = %.14f\n", pi_par);
    printf("Ref estimate of pi = %.14f\n", 4.0 * atan(1.0));
    printf("Sequential code time: %f \n", seq_elapsed);
    printf("Parallel code time: %f \n", par_elapsed);

    if (abs(pi_par - pi_seq) > ACCURACY)
    {
        printf("Test FAILED\n");
    }
    else
    {
        printf("Test PASSED\n");
    }

    return 0;
}

double pi_calc(long long n, long long i, double factor, double sum)
{

    // printf("Before for loop, factor = %f.\n", factor);
    for (i = 0; i < n; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor / (2 * i + 1);
    }
    // printf("After for loop, factor = %f.\n", factor);

    sum = 4.0 * sum;

    return sum;
}

double pi_calc_parallel(long long n, long long i, double factor, double sum)
{

    long long m = n / NUM_TASKS;

#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, factor) \
    shared(n, m, sum)
    {

#pragma omp single
        {

            for (i = 0; i <= NUM_TASKS; i++)
            {
#pragma omp task
                {
                    double mySum = 0.0;
                    for (long long j = 0; j < m; j++)
                    {
                        long long index = i * m + j;
                        if (index == n)
                            break;
                        factor = (index % 2 == 0) ? 1.0 : -1.0;
                        mySum += factor / (2 * index + 1);
                                        }
#pragma omp atomic
                    sum += mySum;
                }
            }
        }
    }

    sum = 4.0 * sum;
    return sum;
}

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
    fprintf(stderr, "   n is the number of terms and should be >= 1\n");
    exit(0);
}
