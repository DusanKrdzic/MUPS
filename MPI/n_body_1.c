
#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"

#define ACCURACY 0.01
#define MASTER 0
#define DIM 2 /* Two-dimensional system */
#define X 0   /* x-coordinate subscript */
#define Y 1   /* y-coordinate subscript */
#define TAG_PARTICLE 1
const double G = 6.673e-11;

typedef double vect_t[DIM]; /* Vector type for position, etc. */

double KE_par, PE_par, KE_seq, PE_seq;

struct thread_particle_s
{
    vect_t *forces;
};

struct particle_s
{
    double m; /* Mass     */
    vect_t s; /* Position */
    vect_t v; /* Velocity */
};

struct particle_s *PS_par, *PS_seq;

void Usage(char *prog_name);
void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,
              double *delta_t_p, int *output_freq_p, char *g_i_p);
// void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Gen_init_cond_par(struct particle_s curr[], int n, int rank, int size);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_my_force(int part, vect_t forces[], struct particle_s curr[], int n);
void Compute_received_force(vect_t *currentForce, vect_t forces[], struct particle_s curr, struct particle_s curr_received[], int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t);
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p);
void n_body_seq(int argc, char *argv[]);
void n_body_par(int argc, char *argv[]);
int N;

int main(int argc, char *argv[])
{

    double finish_seq, start_seq, finish_par, start_par;
    int rank, size;
    // printf("SEQ:\n");

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER)
    {
        GET_TIME(start_seq);

        n_body_seq(argc, argv);
        GET_TIME(finish_seq);

        GET_TIME(start_par);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    n_body_par(argc, argv);

    if (rank == MASTER)
    {
        GET_TIME(finish_par);

        printf("Elapsed time SEQUENTIAL = %f seconds\n", finish_seq - start_seq);
        printf("Elapsed time PARALLEL = %f seconds\n", finish_par - start_par);

        int ok = 1;

        for (int i = 0; i < N; i++)
        {
            if (abs(PS_par[i].m - PS_seq[i].m) > ACCURACY || abs(PS_par[i].s[X] - PS_seq[i].s[X]) > ACCURACY || abs(PS_par[i].s[Y] - PS_seq[i].s[Y]) > ACCURACY || abs(PS_par[i].v[X] - PS_seq[i].v[X]) > ACCURACY || abs(PS_par[i].v[Y] - PS_seq[i].v[Y]) > ACCURACY)
                ok = 0;
        }

        if ((abs(KE_par - KE_seq) < ACCURACY) && (abs(PE_seq - PE_par) < ACCURACY) && ok)
        {
            printf("Test PASSED!\n");
        }
        else
            printf("Test FAILED!\n");
        free(PS_seq);
        free(PS_par);
    }

    MPI_Finalize();

    return 0;

} /* main */

void n_body_seq(int argc, char *argv[])
{
    int n;                   /* Number of particles        */
    int n_steps;             /* Number of timesteps        */
    int step;                /* Current step               */
    int part;                /* Current particle           */
    int output_freq;         /* Frequency of output        */
    double delta_t;          /* Size of timestep           */
    double t;                /* Current Time               */
    struct particle_s *curr; /* Current state of system    */
    vect_t *forces;          /* Forces on each particle    */
    char g_i;                /*_G_en or _i_nput init conds */
                             /* For timings                */

    Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
    curr = malloc(n * sizeof(struct particle_s));
    forces = malloc(n * sizeof(vect_t));

    Gen_init_cond(curr, n);
    N = n;
    double kinetic_energy, potential_energy;
    double start_seq, finish_seq, start_par, finish_par;

    Compute_energy(curr, n, &kinetic_energy, &potential_energy);

    for (step = 1; step <= n_steps; step++)
    {
        t = step * delta_t;
        memset(forces, 0, n * sizeof(vect_t));
        for (part = 0; part < n - 1; part++)
            Compute_my_force(part, forces, curr, n);
        for (part = 0; part < n; part++)
            Update_part(part, forces, curr, n, delta_t);
    }

    Compute_energy(curr, n, &kinetic_energy, &potential_energy);

    PE_seq = potential_energy;
    KE_seq = kinetic_energy;
    PS_seq = curr;

    free(forces);
}

void n_body_par(int argc, char *argv[])
{

    int n;                                   /* Number of particles        */
    int n_steps;                             /* Number of timesteps        */
    int step;                                /* Current step               */
    int part;                                /* Current particle           */
    int output_freq;                         /* Frequency of output        */
    double delta_t;                          /* Size of timestep           */
    double t;                                /* Current Time               */
    struct particle_s *curr, *curr_received; /* Current state of system    */
    vect_t *forces, *forces_received;        /* Forces on each particle    */
    char g_i;                                /*_G_en or _i_nput init conds */
                                             /* For timings                */
    int size;
    int rank;
    MPI_Status status;
    Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunkSize = n / size;
    int chunkMod = n % size;

    int realChunk;

    if (rank < size - 1)
    {
        curr = malloc(chunkSize * sizeof(struct particle_s));
        forces = malloc(chunkSize * sizeof(vect_t));
        realChunk = chunkSize;
    }
    else
    {
        curr = malloc((chunkSize + chunkMod) * sizeof(struct particle_s));
        forces = malloc((chunkSize + chunkMod) * sizeof(vect_t));
        realChunk = chunkSize + chunkMod;
    }

    curr_received = malloc((chunkSize + chunkMod) * sizeof(struct particle_s));
    forces_received = malloc((chunkSize + chunkMod) * sizeof(struct particle_s));

    Gen_init_cond_par(curr, realChunk, rank, size);

    double kinetic_energy, potential_energy;

    MPI_Datatype particletype;
    MPI_Datatype oldtypes[1];
    int blocklens[1];

    MPI_Aint offsets[1];

    offsets[0] = 0;
    oldtypes[0] = MPI_DOUBLE;
    blocklens[0] = 5;

    MPI_Type_create_struct(1, blocklens, offsets, oldtypes, &particletype);
    MPI_Type_commit(&particletype);

    MPI_Datatype forcetype;

    offsets[0] = 0;
    oldtypes[0] = MPI_DOUBLE;
    blocklens[0] = 2;

    MPI_Type_create_struct(1, blocklens, offsets, oldtypes, &forcetype);
    MPI_Type_commit(&forcetype);

    for (step = 1; step <= n_steps; step++)
    {
        t = step * delta_t;
        memset(forces, 0, realChunk * sizeof(vect_t));

        for (int i = rank - 1; i >= 0; i--)
        {

            MPI_Send(curr, realChunk, particletype, i, TAG_PARTICLE, MPI_COMM_WORLD);
        }

        // izracunaj svoje sledece stanje ono sto mozes od 0 do chunkSize-1
        if (rank < size - 1)
        {
            for (part = 0; part < realChunk - 1; part++)
                Compute_my_force(part, forces, curr, realChunk);
        }
        else
        {
            for (part = 0; part < realChunk; part++)
                Compute_my_force(part, forces, curr, realChunk);
        }

        if (rank < size - 1)
        {

            for (int i = rank + 1; i < size; i++)
            {

                MPI_Recv(curr_received, chunkSize + chunkMod, particletype, i, TAG_PARTICLE, MPI_COMM_WORLD, &status);

                memset(forces_received, 0, (chunkSize + chunkMod) * sizeof(vect_t));

                if (i == size - 1)
                {
                    for (int part = 0; part < chunkSize; part++)
                        Compute_received_force(&(forces[part]), forces_received, curr[part], curr_received, (chunkMod + chunkSize));
                    MPI_Send(forces_received, chunkSize + chunkMod, forcetype, i, 2, MPI_COMM_WORLD);
                }
                else
                {
                    for (int part = 0; part < chunkSize; part++)
                        Compute_received_force(&(forces[part]), forces_received, curr[part], curr_received, chunkSize);
                    MPI_Send(forces_received, chunkSize, forcetype, i, 2, MPI_COMM_WORLD);
                }
            }
        }

        for (int i = 0; i < rank; i++)
        {

            MPI_Recv(forces_received, realChunk, forcetype, i, 2, MPI_COMM_WORLD, &status);

            for (int k = 0; k < realChunk; k++)
            {
                forces[k][X] += forces_received[k][X];
                forces[k][Y] += forces_received[k][Y];
            }
        }

        for (part = 0; part < realChunk; part++)
            Update_part(part, forces, curr, realChunk, delta_t);
    }
    if (rank != MASTER)
    {

        MPI_Send(curr, realChunk, particletype, MASTER, 3, MPI_COMM_WORLD);
    }

    if (rank == MASTER)
    {

        PS_par = malloc(n * sizeof(struct particle_s));

        for (int i = 0; i < realChunk; i++)
            PS_par[i] = curr[i];

        for (int i = 1; i < size; i++)
        {
            int currentSize;
            if (i == size - 1)
                currentSize = chunkSize + chunkMod; // ako primam od poslednjeg procesa
            else
                currentSize = chunkSize;

            MPI_Recv((PS_par + chunkSize * i), currentSize, particletype, i, 3, MPI_COMM_WORLD, &status);
        }

        Compute_energy(PS_par, n, &kinetic_energy, &potential_energy);

        PE_par = potential_energy;
        KE_par = kinetic_energy;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(forces);
    free(forces_received);
    free(curr_received);
    MPI_Type_free(&particletype);
    MPI_Type_free(&forcetype);
    free(curr);
}

void Compute_my_force(int part, vect_t forces[], struct particle_s curr[], int n)
{
    int k;
    double mg;
    vect_t f_part_k;
    double len, len_3, fact;

    for (k = part + 1; k < n; k++)
    {
        f_part_k[X] = curr[part].s[X] - curr[k].s[X];
        f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
        len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
        len_3 = len * len * len;
        mg = -G * curr[part].m * curr[k].m;
        fact = mg / len_3;
        f_part_k[X] *= fact;
        f_part_k[Y] *= fact;

        forces[part][X] += f_part_k[X];
        forces[part][Y] += f_part_k[Y];
        forces[k][X] -= f_part_k[X];
        forces[k][Y] -= f_part_k[Y];
    }
} /* Compute_force */

void Compute_received_force(vect_t *currentForce, vect_t forces[], struct particle_s curr, struct particle_s curr_received[], int n)
{
    int k;
    double mg;
    vect_t f_part_k;
    double len, len_3, fact;

    for (k = 0; k < n; k++)
    {
        f_part_k[X] = curr.s[X] - curr_received[k].s[X];
        f_part_k[Y] = curr.s[Y] - curr_received[k].s[Y];
        len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
        len_3 = len * len * len;
        mg = -G * curr.m * curr_received[k].m;
        fact = mg / len_3;
        f_part_k[X] *= fact;
        f_part_k[Y] *= fact;

        (*currentForce)[X] += f_part_k[X];
        (*currentForce)[Y] += f_part_k[Y];

        forces[k][X] -= f_part_k[X];
        forces[k][Y] -= f_part_k[Y];
    }
} /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t)
{
    double fact = delta_t / curr[part].m;

    curr[part].s[X] += delta_t * curr[part].v[X];
    curr[part].s[Y] += delta_t * curr[part].v[Y];
    curr[part].v[X] += fact * forces[part][X];
    curr[part].v[Y] += fact * forces[part][Y];
} /* Update_part */

void Compute_energy(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p)
{
    int i, j;
    vect_t diff;
    double pe = 0.0, ke = 0.0;
    double dist, speed_sqr;

    for (i = 0; i < n; i++)
    {
        speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
        ke += curr[i].m * speed_sqr;
    }
    ke *= 0.5;

    for (i = 0; i < n - 1; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            diff[X] = curr[i].s[X] - curr[j].s[X];
            diff[Y] = curr[i].s[Y] - curr[j].s[Y];
            dist = sqrt(diff[X] * diff[X] + diff[Y] * diff[Y]);
            pe += -G * curr[i].m * curr[j].m / dist;
        }
    }

    *kin_en_p = ke;
    *pot_en_p = pe;
} /* Compute_energy */

void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,
              double *delta_t_p, int *output_freq_p, char *g_i_p)
{
    if (argc != 6)
        Usage(argv[0]);
    *n_p = strtol(argv[1], NULL, 10);
    *n_steps_p = strtol(argv[2], NULL, 10);
    *delta_t_p = strtod(argv[3], NULL);
    *output_freq_p = strtol(argv[4], NULL, 10);
    *g_i_p = argv[5][0];

    if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0)
        Usage(argv[0]);
    if (*g_i_p != 'g' && *g_i_p != 'i')
        Usage(argv[0]);

} /* Get_args */

void Gen_init_cond(struct particle_s curr[], int n)
{
    int part;
    double mass = 5.0e24;
    double gap = 1.0e5;
    double speed = 3.0e4;

    srand(1);

    for (part = 0; part < n; part++)
    {
        curr[part].m = mass;
        curr[part].s[X] = part * gap;
        curr[part].s[Y] = 0.0;
        curr[part].v[X] = 0.0;
        if (part % 2 == 0)
            curr[part].v[Y] = speed;
        else
            curr[part].v[Y] = -speed;
    }
} /* Gen_init_cond */

void Gen_init_cond_par(struct particle_s curr[], int n, int rank, int size)
{

    int part;

    double mass = 5.0e24;

    double gap = 1.0e5;

    double speed = 3.0e4;

    srand(1);

    for (part = 0; part < n; part++)
    {
        curr[part].m = mass;
        curr[part].s[X] = (part + (rank * size)) * gap;
        curr[part].s[Y] = 0.0;
        curr[part].v[X] = 0.0;
        if (part % 2 == 0)
            curr[part].v[Y] = speed;
        else
            curr[part].v[Y] = -speed;
    }
} /* Gen_init_cond */

void Output_state(double time, struct particle_s curr[], int n)
{
    int part;
    printf("%.2f\n", time);
    for (part = 0; part < n; part++)
    {
        printf("%3d %10.3e ", part, curr[part].s[X]);
        printf("  %10.3e ", curr[part].s[Y]);
        printf("  %10.3e ", curr[part].v[X]);
        printf("  %10.3e\n", curr[part].v[Y]);
    }
    printf("\n");
} /* Output_state */

void Usage(char *prog_name)
{
    fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n", prog_name);
    fprintf(stderr, "   <size of timestep> <output frequency>\n");
    fprintf(stderr, "   <g|i>\n");
    fprintf(stderr, "   'g': program should generate init conds\n");
    fprintf(stderr, "   'i': program should get init conds from stdin\n");

    exit(0);
} /* Usage */
