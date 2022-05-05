#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double pi_calc_seq(long long n, long long i, double factor, double sum);
double pi_calc_par(long long n, long long i, double factor, double sum, int size, int rank);

#define MASTER 0
#define ACCURACY 0.01

void Usage(char *prog_name);

int main(int argc, char *argv[])
{
    long long n, i;
    double factor;
    double sum = 0.0;

    int rank, size;
    double par_time1, par_elapsed;
    double seq_time1, seq_elapsed;
    double pi_par, pi_seq;
    long long start, end, chunk;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER)
    {

        if (argc != 2)
            Usage(argv[0]);
        n = strtoll(argv[1], NULL, 10);
        if (n < 1)
            Usage(argv[0]);

        par_time1 = MPI_Wtime();

        for (int i = 0; i < size; i++)
        {
            long long start, end, chunk;
            chunk = (n + size - 1) / size;
            start = rank * chunk;
            end = start + chunk < n ? start + chunk : n;
            MPI_Send(&start, 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD);
            MPI_Send(&end, 1, MPI_LONG_LONG_INT, i, i, MPI_COMM_WORLD);
        }
    }

    MPI_Bcast(&factor, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    pi_par = pi_calc_par(n, i, factor, sum, size, rank);

    if (rank == MASTER)
    {

        par_elapsed = MPI_Wtime() - par_time1;

        seq_time1 = MPI_Wtime();

        sum = 0.0;

        pi_seq = pi_calc_seq(n, i, factor, sum);

        seq_elapsed = MPI_Wtime() - seq_time1;

        printf("Number of processes: %d\n", size);
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
    }

    MPI_Finalize();

    return 0;
}

double pi_calc_par(long long n, long long i, double factor, double sum, int size, int rank)
{

    double sumReduced;

    long long start, end;
    MPI_Status status;
    MPI_Recv(&start, 1, MPI_LONG_LONG_INT, 0, rank, MPI_COMM_WORLD, &status);
    MPI_Recv(&end, 1, MPI_LONG_LONG_INT, 0, rank, MPI_COMM_WORLD, &status);

    for (i = start; i < end; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor / (2 * i + 1);
    }

    MPI_Reduce(&sum, &sumReduced, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER)
        sum = 4.0 * sum;

    return sum;
}

double pi_calc_seq(long long n, long long i, double factor, double sum)
{

    for (i = 0; i < n; i++)
    {
        factor = (i % 2 == 0) ? 1.0 : -1.0;
        sum += factor / (2 * i + 1);
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
