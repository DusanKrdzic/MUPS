
#include "mpi.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define LIMIT -999
#define MASTER 0

void runTest(int argc, char **argv);
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
	int max_rows, max_cols, penalty, idx, index;
	int *input_itemsets, *referrence;

	double time1_seq, time2_seq, elapsed_seq;

	double time1_par, time2_par, elapsed_par;
	int traceback_count_seq;
	int traceback_count_par;
	int rank, size;
	int *traceback_seq;
	int *traceback_par;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size <= 1)
	{
		printf("Can't be less then 2 processes!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	if (argc == 3)
	{
		max_cols = max_rows = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
	else
	{
		usage(argc, argv);
	}

	int colsChunk = max_cols / size;

	MPI_Datatype vector;

	MPI_Type_contiguous(max_rows * (colsChunk + 1), MPI_INT, &vector);
	// colsChunk+1 jer jer su polja prve kolone zapravo polja koja obradjuje prethodni proces
	// i zato 1 kolona vise kako bi dohvatio poslednje polje prethodnog procesa u tekucem redu
	// koje je potrebno za racunanje prvog polja tekuceg procesa u tekucem redu..
	//+colsChunk su kolone koje zapravo obradjuje tekuci proces
	MPI_Type_commit(&vector);

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	referrence = (int *)malloc(max_rows * max_cols * sizeof(int));

	input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

	double time_seq, time_par;
	double time_start, time_end;

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

	for (int i = 0; i < max_cols; i++)
	{
		for (int j = 0; j < max_rows; j++)
		{
			input_itemsets[i * max_cols + j] = 0;
		}
	}

	for (int i = 1; i < max_rows; i++)
	{
		input_itemsets[i * max_cols] = i % 10 + 1;
	}
	for (int j = 1; j < max_cols; j++)
	{
		input_itemsets[j] = j % 10 + 1;
	}

	for (int i = 1; i < max_cols; i++)
	{
		for (int j = 1; j < max_rows; j++)
		{
			referrence[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
		}
	}
	// sekvencijalni deo
	if (rank == MASTER)
	{
		for (int i = 1; i < max_rows; i++)
			input_itemsets[i * max_cols] = -i * penalty;
		for (int j = 1; j < max_cols; j++)
			input_itemsets[j] = -j * penalty;
		time_start = MPI_Wtime();
		// printf("Processing top-left matrix\n");
		for (int i = 0; i < max_cols - 2; i++)
		{
			for (idx = 0; idx <= i; idx++)
			{
				index = (idx + 1) * max_cols + (i + 1 - idx);
				input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
												input_itemsets[index - 1] - penalty,
												input_itemsets[index - max_cols] - penalty);
			}
		}

		// printf("Processing bottom-right matrix\n");
		for (int i = max_cols - 4; i >= 0; i--)
		{
			for (idx = 0; idx <= i; idx++)
			{
				index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
				input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + referrence[index],
												input_itemsets[index - 1] - penalty,
												input_itemsets[index - max_cols] - penalty);
			}
		}

		time_end = MPI_Wtime();
		time_seq = time_end - time_start;

#define TRACEBACK
#ifdef TRACEBACK

		FILE *fpo;
		int *result = (int *)malloc(max_rows * max_cols * sizeof(int));
		int tek = 0;

		fpo = fopen("result_seq.txt", "w");

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

		traceback_seq = result;
		traceback_count_seq = tek;

#endif
	}
	// paralelni deo
	if (rank == MASTER)
	{

		for (int i = 0; i < max_cols; i++)
		{
			for (int j = 0; j < max_rows; j++)
			{
				input_itemsets[i * max_cols + j] = 0;
			}
		}

		for (int i = 1; i < max_rows; i++)
			input_itemsets[i * max_cols] = -i * penalty;
		for (int j = 1; j < max_cols; j++)
			input_itemsets[j] = -j * penalty;

		time_start = MPI_Wtime();
	}
	else
	{

		free(input_itemsets);
		input_itemsets = (int *)malloc(max_rows * (colsChunk + 1) * sizeof(int));

		for (int i = 0; i < max_rows; i++)
		{
			for (int j = 0; j < colsChunk + 1; j++)
			{
				input_itemsets[i * (colsChunk + 1) + j] = 0;
			}
		}
		for (int i = 0; i < colsChunk + 1; i++)
			input_itemsets[i] = -i * penalty - rank * colsChunk * penalty;
	}
	for (int row = 1; row < max_rows; row++)
	{ // obradi sve redove, popunjavanje po kolonama (od 1 jer 0. red ima penalty-je koji su vec izracunati)

		if (rank == MASTER)
		{
			for (int col = 1; col < colsChunk + 1; col++)
			{ // obradi sve svoje kolone tekuceg reda
				// MASTER nema proces i-1 od kog bi cekao polje, tj nema zavisnost.
				// MASTER odmah ide da racuna polja svih kolona tekuceg reda
				input_itemsets[row * max_cols + col] = maximum(input_itemsets[(row - 1) * max_cols + col - 1] + referrence[row * max_cols + col],
															   input_itemsets[row * max_cols + col - 1] - penalty,
															   input_itemsets[(row - 1) * max_cols + col - 1] - penalty);

				if (col == colsChunk)
				{ // ako smo stigli do poslednje kolone u tekucem redu posalji to polje sledecm procesu

					MPI_Send(&input_itemsets[row * max_cols + col], 1, MPI_INT, rank + 1, 100, MPI_COMM_WORLD);
				}
			}
		}
		else

		{
			for (int col = 0; col < colsChunk + 1; col++)
			{ // ostali procesi obradite sve svoje kolone tekuceg reda.

				if (col == 0)
				{ // ako radite prvu kolonu tekuceg reda, to je zapravo polje koje obradjuje
					// prethodni proces, i treba da primim to polje od prethodnog procesa
					// kako bih moga da izracunam moje prvo polje (kolonu) u tekucem redu
					MPI_Recv(&input_itemsets[row * (colsChunk + 1) + col], 1, MPI_INT, rank - 1, 100, MPI_COMM_WORLD, &status);
				}
				else
				{ // moja polja koja treba da izracunam
					input_itemsets[row * (colsChunk + 1) + col] = maximum(input_itemsets[(row - 1) * (colsChunk + 1) + col - 1] + referrence[row * max_rows + colsChunk * rank + col],
																		  input_itemsets[row * (colsChunk + 1) + col - 1] - penalty,
																		  input_itemsets[(row - 1) * (colsChunk + 1) + col - 1] - penalty);
					if (rank < size - 1 && col == colsChunk)
						MPI_Send(&input_itemsets[row * (colsChunk + 1) + col], 1, MPI_INT, rank + 1, 100, MPI_COMM_WORLD);
				}
			}

			if (row == max_rows - 1) // ako sam obradio sve posalji masteru da skupi
				MPI_Send(input_itemsets, 1, vector, MASTER, 100, MPI_COMM_WORLD);
		}
	}

	if (rank == MASTER)
	{ // master da prikupi od svih

		int *buffer = (int *)malloc(max_rows * (colsChunk + 1) * sizeof(int));

		for (int proc = 1; proc < size; proc++)
		{
			MPI_Recv(buffer, max_rows * (colsChunk + 1), MPI_INT, proc, 100, MPI_COMM_WORLD, &status);

			int begin = colsChunk * proc;
			int end = colsChunk * (proc + 1);
			int num = colsChunk;
			for (int i = 1; i < max_rows; i++)
			{
				for (int j = begin + 1; j < end + 1; j++)
				{
					num++;
					if (num % (colsChunk + 1) == 0)
						num++;
					input_itemsets[i * max_rows + j] = buffer[num];
				}
			}
		}

		time_end = MPI_Wtime();
		time_par = time_end - time_start;
		printf("Number of processes:%d\n", size);
		printf("Sequential time: %f.\n", time_seq);
		printf("Parallel time: %f.\n", time_par);

#define TRACEBACK
#ifdef TRACEBACK

		FILE *fpo;
		int *result = (int *)malloc(max_rows * max_cols * sizeof(int));
		int tek = 0;

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

		traceback_par = result;
		traceback_count_par = tek;
#endif

		int fail = 0;
		if (traceback_count_par != traceback_count_seq)
		{
			fail = 1;
		}
		else
		{
			for (int i = 0; i < traceback_count_par; i++)
			{

				if (traceback_par[i] != traceback_seq[i])
				{
					fail = 1;
					break;
				}
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
	}

	free(referrence);
	free(input_itemsets);
	if (rank == MASTER)
	{
		free(traceback_par);
		free(traceback_seq);
	}
	MPI_Finalize();
}
