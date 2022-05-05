#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "timer.h"
#include <omp.h>
#define ACCURACY 0.01
#define DIM 2 /* Two-dimensional system */
#define X 0   /* x-coordinate subscript */
#define Y 1   /* y-coordinate subscript */
#define NUM_THREADS 8
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
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force_seq(int part, vect_t forces[], struct particle_s curr[], int n);
void Compute_force_par(int part, vect_t forces[], struct particle_s curr[], int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t);
void Compute_energy_par(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p);
void Compute_energy_seq(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p);
void n_body_seq(int argc, char *argv[]);
void n_body_par(int argc, char *argv[]);
int N;

int main(int argc, char *argv[])
{

   double finish_seq, start_seq, finish_par, start_par;

   // printf("SEQ:\n");
   GET_TIME(start_seq);
   // start_seq=omp_get_wtime();
   n_body_seq(argc, argv);
   GET_TIME(finish_seq);
   // finish_seq=omp_get_wtime();
   // printf("PAR:\n");
   GET_TIME(start_par);
   // start_par=omp_get_wtime();
   n_body_par(argc, argv);
   GET_TIME(finish_par);
   // finish_par=omp_get_wtime();

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
   /*
   if (g_i == 'i')
      Get_init_cond(curr, n);
   else
   */
   Gen_init_cond(curr, n);
   N = n;
   double kinetic_energy, potential_energy;
   double start_seq, finish_seq, start_par, finish_par;

   printf("Threads in parallel code: %d\nNumber of particles:%d\n", NUM_THREADS, n);

   Compute_energy_seq(curr, n, &kinetic_energy, &potential_energy);
   // printf("ENERGY BEFORE:  PE = %e, KE = %e, Total Energy = %e\n", potential_energy, kinetic_energy, kinetic_energy + potential_energy);
   // Output_state(0, curr, n);
   for (step = 1; step <= n_steps; step++)
   {
      t = step * delta_t;
      memset(forces, 0, n * sizeof(vect_t));
      for (part = 0; part < n - 1; part++)
         Compute_force_seq(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
   }
   // Output_state(t, curr, n);
   Compute_energy_seq(curr, n, &kinetic_energy, &potential_energy);
   // printf("ENERGY AFTER:   PE = %e, KE = %e, Total Energy = %e\n", potential_energy, kinetic_energy, kinetic_energy + potential_energy);
   PE_seq = potential_energy;
   KE_seq = kinetic_energy;
   PS_seq = curr;

   free(forces);
}

void n_body_par(int argc, char *argv[])
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

   /*
      if (g_i == 'i')
         Get_init_cond(curr, n);
      else
      */
   Gen_init_cond(curr, n);

   double kinetic_energy, potential_energy;

   Compute_energy_par(curr, n, &kinetic_energy, &potential_energy);
   // printf("ENERGY BEFORE:   PE = %e, KE = %e, Total Energy = %e\n", potential_energy, kinetic_energy, kinetic_energy + potential_energy);
   // Output_state(0, curr, n);
   for (step = 1; step <= n_steps; step++)
   {
      t = step * delta_t;
      memset(forces, 0, n * sizeof(vect_t));

#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, 10)
      for (part = 0; part < n - 1; part++)
         Compute_force_par(part, forces, curr, n);

#pragma omp parallel for num_threads(NUM_THREADS)
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
   }
   // Output_state(t, curr, n);
   Compute_energy_par(curr, n, &kinetic_energy, &potential_energy);
   // printf("ENERGY AFTER:   PE = %e, KE = %e, Total Energy = %e\n", potential_energy, kinetic_energy, kinetic_energy + potential_energy);

   PE_par = potential_energy;
   KE_par = kinetic_energy;
   PS_par = curr;

   free(forces);
}

void Compute_force_seq(int part, vect_t forces[], struct particle_s curr[], int n)
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

void Compute_force_par(int part, vect_t forces[], struct particle_s curr[], int n)
{

   int k;
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;

   double sum_x = 0.0, sum_y = 0.0;

   for (k = part + 1; k < n; k++)
   {

      // rastojanje izmeju cestice part i k.. len=sqrt((x_part-x_k)^2+(y_part-y_k)^2)
      f_part_k[X] = curr[part].s[X] - curr[k].s[X];
      f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
      len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
      // imenilac u fomuli (len^3)
      len_3 = len * len * len;
      // masa
      mg = -G * curr[part].m * curr[k].m;
      // konacni factor za vektor
      fact = mg / len_3;

      f_part_k[X] *= fact;
      f_part_k[Y] *= fact;

      sum_x += f_part_k[X];
      sum_y += f_part_k[Y];

#pragma omp atomic
      forces[k][X] -= f_part_k[X];
#pragma omp atomic
      forces[k][Y] -= f_part_k[Y];
   }

#pragma omp atomic
   forces[part][X] += sum_x;
#pragma omp atomic
   forces[part][Y] += sum_y;

} /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t)
{
   double fact = delta_t / curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
} /* Update_part */

void Compute_energy_seq(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p)
{
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;
   // racuna kineticku energiju za svaku cesticu kao ke=m*(vx^2+vy^2)
   for (i = 0; i < n; i++)
   {
      speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
      ke += curr[i].m * speed_sqr;
   }
   ke *= 0.5;
   // racuna potencijalnu energiju
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

void Compute_energy_par(struct particle_s curr[], int n, double *kin_en_p,
                        double *pot_en_p)
{
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;
// racuna kineticku energiju za svaku cesticu kao ke=m*(vx^2+vy^2)
#pragma omp parallel for num_threads(NUM_THREADS) default(none) private(speed_sqr, i) reduction(+ \
                                                                                                : ke) shared(curr, n)
   for (i = 0; i < n; i++)
   {
      speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
      ke += curr[i].m * speed_sqr;
   }
   ke *= 0.5;
// racuna potencijalnu energiju
#pragma omp parallel for num_threads(NUM_THREADS) default(none) private(diff, i, j, dist) reduction(+ \
                                                                                                    : pe) shared(curr, n)
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
/*
void Get_init_cond(struct particle_s curr[], int n)
{
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++)
   {
      scanf("%lf", &curr[part].m);
      scanf("%lf", &curr[part].s[X]);
      scanf("%lf", &curr[part].s[Y]);
      scanf("%lf", &curr[part].v[X]);
      scanf("%lf", &curr[part].v[Y]);
   }
} /* Get_init_cond */

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
