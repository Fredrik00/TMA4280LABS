/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define PI 3.14159265358979323846
#define true 1
#define false 0

typedef double real;
typedef int bool;

// Function prototypes
real *mk_1D_array(size_t n, bool zero);
int *mk_int_array(size_t n, bool zero);
real **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(real **bt, real **b, size_t m);
real rhs(real x, real y);
void distribute_work(int total, int *work, int *recvcounts, int *recvdisps);

// Functions implemented in FORTRAN in fst.f and called from C.
// The trailing underscore comes from a convention for symbol names, called name
// mangling: if can differ with compilers.
void fst_(real *v, int *n, real *w, int *nn);
void fstinv_(real *v, int *n, real *w, int *nn);

int main(int argc, char **argv)
{
	if (argc < 2) {
	printf("Usage:\n");
	printf("  poisson n\n\n");
	printf("Arguments:\n");
	printf("  n: the problem size (must be a power of 2)\n");
	}

	/*
	*  The equation is solved on a 2D structured grid and homogeneous Dirichlet
	*  conditions are applied on the boundary:
	*  - the number of grid points in each direction is n+1,
	*  - the number of degrees of freedom in each direction is m = n-1,
	*  - the mesh size is constant h = 1/n.
	*/
	int n = atoi(argv[1]);
	int m = n - 1;
	real h = 1.0 / n;

	int nprocs, rank;
	int tag = 100;
	MPI_Status status;
	double time_start;
	double duration;
	// Use these variables globally?
	int *work = mk_int_array(2, true);  // index, number of rows
	int *sendcounts = mk_int_array(nprocs, true);
	int *recvcounts = mk_int_array(nprocs, true);
	int *senddisps = mk_int_array(nprocs, true);
	int *recvdisps = mk_int_array(nprocs, true);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		time_start = MPI_Wtime();	
	}

	/*
	* Grid points are generated with constant mesh size on both x- and y-axis.
	*/
	real *grid = mk_1D_array(n+1, false);
	distribute_work(n+1, work, recvcounts, recvdisps);
	
	for (int i = 0; i < nprocs; i++) { // As the setup of buffers do not scale with n it is not treated by parallellization
		sendcounts[i] = work[1];
		senddisps[i] = work[0];
	}
	
	#pragma omp parallel for schedule(static)
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		grid[i] = i * h;
	}

	MPI_Alltoallv(grid, sendcounts, senddisps, MPI_DOUBLE, grid, recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	/*
	* The diagonal of the eigenvalue matrix of T is set with the eigenvalues
	* defined Chapter 9. page 93 of the Lecture Notes.
	* Note that the indexing starts from zero here, thus i+1.
	*/
	real *diag = mk_1D_array(m, false);
	distribute_work(m, work, recvcounts, recvdisps);
	
	for (int i = 0; i < nprocs; i++) {
		sendcounts[i] = work[1];
		senddisps[i] = work[0];
	}
	
	#pragma omp parallel for schedule(static)
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n));
	}

	MPI_Alltoallv(diag, sendcounts, senddisps, MPI_DOUBLE, diag, recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	/*
	* Allocate the matrices b and bt which will be used for storing value of
	* G, \tilde G^T, \tilde U^T, U as described in Chapter 9. page 101.
	*/
	real **b = mk_2D_array(m, m, false);
	real **bt = mk_2D_array(m, m, false);

	/*
	* This vector will holds coefficients of the Discrete Sine Transform (DST)
	* but also of the Fast Fourier Transform used in the FORTRAN code.
	* The storage size is set to nn = 4 * n, look at Chapter 9. pages 98-100:
	* - Fourier coefficients are complex so storage is used for the real part
	*   and the imaginary part.
	* - Fourier coefficients are defined for j = [[ - (n-1), + (n-1) ]] while 
	*   DST coefficients are defined for j [[ 0, n-1 ]].
	* As explained in the Lecture notes coefficients for positive j are stored
	* first.
	* The array is allocated once and passed as arguments to avoid doings 
	* reallocations at each function call.
	*/
	int threads = omp_get_num_threads();
	int nn = 4 * n;
	//real *z = mk_1D_array(nn, false);
	real **z = mk_2D_array(threads, nn, false);
	
	/*
	* Initialize the right hand side data for a given rhs function.
	* Note that the right hand-side is set at nodes corresponding to degrees
	* of freedom, so it excludes the boundary (bug fixed by petterjf 2017).
	* 
	*/

	// Work already distributed for m
	for (int i = 0; i < nprocs; i++) {
		sendcounts[i] = work[1]*m;
		senddisps[i] = work[0]*m;
		recvcounts[i] = recvcounts[i]*m;
		recvdisps[i] = recvdisps[i]*m;
	}
	
	#pragma omp parallel for schedule(static) // Inner loop, outer or both?
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		for (size_t j = 0; j < m; j++) {
			b[i][j] = h * h * rhs(grid[i+1], grid[j+1]);
		}
	}

	
	MPI_Alltoallv(b[0], sendcounts, senddisps, MPI_DOUBLE, b[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	/*
	* Compute \tilde G^T = S^-1 * (S * G)^T (Chapter 9. page 101 step 1)
	* Instead of using two matrix-matrix products the Discrete Sine Transform
	* (DST) is used.
	* The DST code is implemented in FORTRAN in fsf.f and can be called from C.
	* The array zz is used as storage for DST coefficients and internally for 
	* FFT coefficients in fst_ and fstinv_.
	* In functions fst_ and fst_inv_ coefficients are written back to the input 
	* array (first argument) so that the initial values are overwritten.
	*/
	// Work already distributed for m
	// Memory bound? Perhaps create local array segments and add up afterwards

	#pragma omp for schedule(static)  // Parallel modifier caused errors
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		fst_(b[i], &n, z[omp_get_thread_num()], &nn);
	}
	
	MPI_Alltoallv(b[0], sendcounts, senddisps, MPI_DOUBLE, b[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);
	
	transpose(bt, b, m);

	// Work already distributed for m
	#pragma omp for schedule(static)  // Parallel modifier caused errors
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		fstinv_(bt[i], &n, z[omp_get_thread_num()], &nn);
	}
	
	MPI_Alltoallv(bt[0], sendcounts, senddisps, MPI_DOUBLE, bt[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);
	
	/*
	* Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
	*/
	// Work already distributed for m
	#pragma omp parallel for schedule(static)
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		for (size_t j = 0; j < m; j++) {
			bt[i][j] = bt[i][j] / (diag[i] + diag[j]);
		}
	}
	
	MPI_Alltoallv(bt[0], sendcounts, senddisps, MPI_DOUBLE, bt[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);
	
	/*
	* Compute U = S^-1 * (S * Utilde^T) (Chapter 9. page 101 step 3)
	*/
	// Work already distributed for m
	#pragma omp for schedule(static)  // Parallel modifier caused error, threads access the same z
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		fst_(bt[i], &n, z[omp_get_thread_num()], &nn);
	}

	MPI_Alltoallv(bt[0], sendcounts, senddisps, MPI_DOUBLE, bt[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	transpose(b, bt, m);
	
	// Work already distributed for m
	#pragma omp for schedule(static)  // Parallel modifier caused errors
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		fstinv_(b[i], &n, z[omp_get_thread_num()], &nn);
	}

	MPI_Alltoallv(b[0], sendcounts, senddisps, MPI_DOUBLE, b[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	/*
	* Compute maximal value of solution for convergence analysis in L_\infty
	* norm.
	*/
	double u_max = 0.0;
	// Work already distributed for m
	#pragma omp parallel for schedule(static)
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		for (size_t j = 0; j < m; j++) {
			u_max = u_max > b[i][j] ? u_max : b[i][j];
		}
	}
	
	MPI_Allreduce(&u_max, &u_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (rank == 0) {
		duration = MPI_Wtime() - time_start;
		printf("duration: %e\n", duration);
		printf("u_max = %e\n", u_max);
	}
	
	MPI_Finalize();

	return 0;
}

/*
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to swtich between problem definitions.
 */

real rhs(real x, real y) {
	return 2 * (y - y*y + x - x*x);
}

/*
 * Write the transpose of b a matrix of R^(m*m) in bt.
 * In parallel the function MPI_Alltoallv is used to map directly the entries
 * stored in the array to the block structure, using displacement arrays.
 */

void transpose(real **bt, real **b, size_t m)
{
	int *rows = mk_int_array(2, true);  // index, number of rows
	int nprocs, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int *sendcounts = mk_int_array(nprocs, true);
	int *recvcounts = mk_int_array(nprocs, true);
	int *senddisps = mk_int_array(nprocs, true);
	int *recvdisps = mk_int_array(nprocs, true);

	if (rank == 0 && m < 2) {  // Printing for lower m to check that the transpose is correct
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < m; j++) {
				printf("%f	", b[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	distribute_work(m, rows, recvcounts, recvdisps);  // might want to handle recv buffers another place	
	//MPI_Barrier(MPI_COMM_WORLD);
	
	for (int i = 0; i < nprocs; i++) {
		sendcounts[i] = rows[1]*m;
		senddisps[i] = rows[0]*m;
		recvcounts[i] = recvcounts[i]*m;
		recvdisps[i] = recvdisps[i]*m;
	}
	
	// Each process allocates their assigned set of transposed rows from the original columns
	#pragma omp parallel for schedule(static)
	for (size_t i = rows[0]; i < rows[0] + rows[1]; i++) {
		for (size_t j = 0; j < m; j++) {
			bt[i][j] = b[j][i];
		}
	}

	// Every process sends their transposed rows, and stores received ones in bt
	MPI_Alltoallv(bt[0], sendcounts, senddisps, MPI_DOUBLE, bt[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	
	if (rank == 0 && m < 2) {
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < m; j++) {
				printf("%f	", bt[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

/*
 * The allocation of a vectore of size n is done with just allocating an array.
 * The only thing to notice here is the use of calloc to zero the array.
 */

real *mk_1D_array(size_t n, bool zero)
{
	if (zero) {
		return (real *)calloc(n, sizeof(real));
	}
	return (real *)malloc(n * sizeof(real));
}

int *mk_int_array(size_t n, bool zero)
{
	if (zero) {
		return (int *)calloc(n, sizeof(int));
	}
	return (int *)malloc(n * sizeof(int));
}

/*
 * The allocation of the two-dimensional array used for storing matrices is done
 * in the following way for a matrix in R^(n1*n2):
 * 1. an array of pointers is allocated, one pointer for each row,
 * 2. a 'flat' array of size n1*n2 is allocated to ensure that the memory space
 *   is contigusous,
 * 3. pointers are set for each row to the address of first element.
 */

real **mk_2D_array(size_t n1, size_t n2, bool zero)
{
	// 1
	real **ret = (real **)malloc(n1 * sizeof(real *));

	// 2
	if (zero) {
		ret[0] = (real *)calloc(n1 * n2, sizeof(real));
	}
	else {
		ret[0] = (real *)malloc(n1 * n2 * sizeof(real));
	}

	// 3
	for (size_t i = 1; i < n1; i++) {
		ret[i] = ret[i-1] + n2;
	}
	return ret;
}

void distribute_work(int total, int *work, int *recvcounts, int *recvdisps) 
{
	int nprocs, rank;
	int tag = 100;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank ==  0) {
		int work_left = total;
		int p_left = nprocs;
	
		while (p_left > 0) {
			int p = nprocs-p_left;
			work[1] = work_left/p_left;
			MPI_Send(work, 2, MPI_INT, p, tag, MPI_COMM_WORLD);
			recvcounts[p] = work[1];
			recvdisps[p] = work[0];
			work[0] += work[1];
			work_left -= work[1];
			p_left--;
		}
	}
	
	MPI_Recv(work, 2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);

	// Every process need a copy of the receive buffer information
	MPI_Bcast(recvcounts, nprocs, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(recvdisps, nprocs, MPI_INT, 0, MPI_COMM_WORLD);
}
