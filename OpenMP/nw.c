#define LIMIT -999
#define NUM_THREADS 8
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void runTest(int argc, char **argv);
void nw_parallel(int *input_itemsets, int *referrence, int max_cols, int max_rows, int penalty);
void nw_sequential(int *input_itemsets, int *referrence, int max_cols, int max_rows, int penalty);
int *traceback(int *, int *, int, int, int, int);

int maximum(int a, int b, int c)
{
    int k;
    if (a <= b)
        k = b;
    else
        k = a;
    if (k <= c)
        return (c);
    else
        return (k);
}

int blosum62[24][24] = {
    {4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
    {-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
    {-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
    {-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
    {-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
    {-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
    {-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
    {-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4},
    {-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4},
    {-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
    {-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4},
    {-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4},
    {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
    {1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
    {0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
    {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
    {-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
    {0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4},
    {-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},
    {-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
    {0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
    {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

double gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

int main(int argc, char **argv)
{
    runTest(argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
    fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
    fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
    exit(1);
}

void runTest(int argc, char **argv)
{

    int max_rows, max_cols, penalty;
    int *input_itemsets_par, *input_itemsets_seq, *referrence;
    int *traceback_seq;
    int *traceback_par;

    if (argc == 3)
    {
        max_cols = max_rows = atoi(argv[1]);
        penalty = atoi(argv[2]);
    }
    else
    {
        usage(argc, argv);
    }

    max_rows = max_rows + 1;
    max_cols = max_cols + 1;
    referrence = (int *)malloc(max_rows * max_cols * sizeof(int));
    input_itemsets_par = (int *)malloc(max_rows * max_cols * sizeof(int));
    input_itemsets_seq = (int *)malloc(max_rows * max_cols * sizeof(int));

    if (!input_itemsets_par)
        fprintf(stderr, "error: can not allocate memory");

    if (!input_itemsets_seq)
        fprintf(stderr, "error: can not allocate memory");

    srand(time(NULL));

    for (int i = 0; i < max_cols; i++)
    {
        for (int j = 0; j < max_rows; j++)
        {
            input_itemsets_seq[i * max_cols + j] = 0;
            input_itemsets_par[i * max_cols + j] = 0;
        }
    }

    //  printf("Start Needleman-Wunsch\n");

    for (int i = 1; i < max_rows; i++)
    {
        input_itemsets_seq[i * max_cols] = rand() % 10 + 1;
        input_itemsets_par[i * max_cols] = input_itemsets_seq[i * max_cols];
    }
    for (int j = 1; j < max_cols; j++)
    {
        input_itemsets_seq[j] = rand() % 10 + 1;
        input_itemsets_par[j] = input_itemsets_seq[j];
    }

    for (int i = 1; i < max_cols; i++)
    {
        for (int j = 1; j < max_rows; j++)
        {
            referrence[i * max_cols + j] = blosum62[input_itemsets_seq[i * max_cols]][input_itemsets_seq[j]];
        }
    }

    for (int i = 1; i < max_rows; i++)
    {
        input_itemsets_seq[i * max_cols] = -i * penalty;
        input_itemsets_par[i * max_cols] = -i * penalty;
    }

    for (int j = 1; j < max_cols; j++)
    {
        input_itemsets_seq[j] = -j * penalty;
        input_itemsets_par[j] = -j * penalty;
    }

    double time_seq, time_par;
    double time_start, time_end;

    time_start = gettime();
    nw_sequential(input_itemsets_seq, referrence, max_cols, max_rows, penalty);
    time_end = gettime();
    time_seq = time_end - time_start;

    time_start = gettime();
    nw_parallel(input_itemsets_par, referrence, max_cols, max_rows, penalty);
    time_end = gettime();
    time_par = time_end - time_start;

    traceback_seq = traceback(input_itemsets_seq, referrence, max_rows, max_cols, penalty, 1);
    traceback_par = traceback(input_itemsets_par, referrence, max_rows, max_cols, penalty, 0);

    printf("Threads in parallel code: %d\nRows and Columns:%d\nPenalty:%d\n", NUM_THREADS, max_rows - 1, penalty);

    printf("Sequential time: %f\n", time_seq);
    printf("Parallel time: %f\n", time_par);

    int fail = 0;
    for (int i = 0; i < max_rows * max_cols; i++)
    {
        if (traceback_par[i] != traceback_seq[i])
        {
            fail = 1;
            break;
        }
    }
    if (fail)
    {
        printf("Test FAILED\n");
    }
    else
    {
        printf("Test PASSED\n");
    }

    free(referrence);
    free(input_itemsets_par);
    free(input_itemsets_seq);
    free(traceback_seq);
    free(traceback_par);
}

void nw_sequential(int *input_itemsets, int *referrence, int max_cols, int max_rows, int penalty)
{
    /*
        for (int i = 1; i < max_rows; i++)
        {
            input_itemsets[i * max_cols] = -i * penalty;
        }

        for (int j = 1; j < max_cols; j++)
        {
            input_itemsets[j] = -j * penalty;
        }
    */
    // printf("Processing top-left matrix\n");
    for (int i = 0; i < max_cols - 2; i++)
    {

        for (int idx = 0; idx <= i; idx++)
        {
            int index = (idx + 1) * max_cols + (i + 1 - idx);
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
        }
    }
    // printf("Processing bottom-right matrix\n");
    for (int i = max_cols - 4; i >= 0; i--)
    {
        for (int idx = 0; idx <= i; idx++)
        {
            int index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
        }
    }
}

void nw_parallel(int *input_itemsets, int *referrence, int max_cols, int max_rows, int penalty)
{

    /*
    #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 1; i < max_rows; i++)
        {
            input_itemsets[i * max_cols] = -i * penalty;
        }

    #pragma omp parallel for num_threads(NUM_THREADS)
        for (int j = 1; j < max_cols; j++)
        {
            input_itemsets[j] = -j * penalty;
        }
    */
    // printf("Processing top-left matrix\n");
    for (int i = 0; i < max_cols - 2; i++)
    {

#pragma omp parallel for num_threads(NUM_THREADS)
        for (int idx = 0; idx <= i; idx++)
        {
            int index = (idx + 1) * max_cols + (i + 1 - idx);
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
        }
    }
    // printf("Processing bottom-right matrix\n");
    for (int i = max_cols - 4; i >= 0; i--)
    {
#pragma omp parallel for num_threads(NUM_THREADS)
        for (int idx = 0; idx <= i; idx++)
        {
            int index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
            input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
                                            input_itemsets[index - 1] - penalty,
                                            input_itemsets[index - max_cols] - penalty);
        }
    }
}

int *traceback(int *input_itemsets, int *referrence, int max_rows, int max_cols, int penalty, int seq)
{

#define TRACEBACK
#ifdef TRACEBACK

    FILE *fpo;
    int *result = (int *)malloc(max_rows * max_cols * sizeof(int));
    int tek = 0;

    if (seq)
        fpo = fopen("result_seq.txt", "w");
    else
        fpo = fopen("result_par.txt", "w");

    fprintf(fpo, "print traceback value:\n");

    for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
    {
        int nw, n, w, traceback;
        if (i == max_rows - 2 && j == max_rows - 2)
        {
            fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]);
            result[tek++] = input_itemsets[i * max_cols + j];
        }
        if (i == 0 && j == 0)
            break;
        if (i > 0 && j > 0)
        {
            nw = input_itemsets[(i - 1) * max_cols + j - 1];
            w = input_itemsets[i * max_cols + j - 1];
            n = input_itemsets[(i - 1) * max_cols + j];
        }
        else if (i == 0)
        {
            nw = n = LIMIT;
            w = input_itemsets[i * max_cols + j - 1];
        }
        else if (j == 0)
        {
            nw = w = LIMIT;
            n = input_itemsets[(i - 1) * max_cols + j];
        }
        else
        {
        }

        // traceback = maximum(nw, w, n);
        int new_nw, new_w, new_n;

        new_nw = nw + referrence[i * max_cols + j];
        new_w = w - penalty;
        new_n = n - penalty;

        traceback = maximum(new_nw, new_w, new_n);
        if (traceback == new_nw)
            traceback = nw;
        if (traceback == new_w)
            traceback = w;
        if (traceback == new_n)
            traceback = n;

        fprintf(fpo, "%d ", traceback);
        result[tek++] = traceback;

        if (traceback == nw)
        {
            i--;
            j--;
            continue;
        }

        else if (traceback == w)
        {
            j--;
            continue;
        }

        else if (traceback == n)
        {
            i--;
            continue;
        }

        else
            ;
    }

    fclose(fpo);

    return result;

#endif
}