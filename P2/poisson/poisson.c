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
#include <string.h>

#define PI 3.14159265358979323846
#define true 1
#define false 0

typedef double real;
typedef int bool;

// Function prototypes
void poisson(int n);
real *mk_1D_array(size_t n, bool zero);
int *mk_int_array(size_t n, bool zero);
real **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(real **bt, real **b, size_t m, int* work, int* senddisps, int* sendcounts, int* recvdisps, int* recvcounts);
real rhs(real x, real y);
real solution(real x, real y);
void distribute_work(int total, int *work, int *recvcounts, int *recvdisps);
void print_matrix(real **matrix, int m);

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

	MPI_Init(&argc, &argv);
	int rank;
	int n;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (strcmp(argv[1], "cvtest") == 0){
		for (int i=2; i<12; i++){
			n = pow(2, i);
			if (rank == 0){ printf("Convergence test	n = %i\n", n); }
			poisson(n);
		}
	}

	else {
		n = atoi(argv[1]);
		poisson(n);
	}
	
	MPI_Finalize();
	return 0;
}

void poisson(int n) {
	/*
	*  The equation is solved on a 2D structured grid and homogeneous Dirichlet
	*  conditions are applied on the boundary:
	*  - the number of grid points in each direction is n+1,
	*  - the number of degrees of freedom in each direction is m = n-1,
	*  - the mesh size is constant h = 1/n.
	*/
	int m = n - 1;
	real h = 1.0 / n;

	int nprocs, rank;
	int tag = 100;
	MPI_Status status;
	double time_start;
	double duration;
	int *work = mk_int_array(2, true);  // index, number of rows
	int *senddisps = mk_int_array(nprocs, false);
	int *sendcounts = mk_int_array(nprocs, false);
	int *recvdisps = mk_int_array(nprocs, false);
	int *recvcounts = mk_int_array(nprocs, false);
	int threads;

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	#pragma omp parallel
	{
		threads = omp_get_num_threads();	
	}

	if (rank == 0) {
		time_start = MPI_Wtime();
		printf("threads %d\n", threads);
	}

	/*
	* Grid points are generated with constant mesh size on both x- and y-axis.
	*/
	real *grid = mk_1D_array(n+1, false);
	distribute_work(n+1, work, recvdisps, recvcounts);
	
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
	distribute_work(m, work, recvdisps, recvcounts);
	
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
	
	transpose(bt, b, m, work, senddisps, sendcounts, recvdisps, recvcounts);

	// Work already distributed for m
	#pragma omp for schedule(static)  // Parallel modifier caused errors
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		//printf("%d\n", omp_get_thread_num());
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

	transpose(b, bt, m, work, senddisps, sendcounts, recvdisps, recvcounts);
	
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
	// Add openmp, accessing same u_max
	#pragma omp parallel for reduction(max : u_max)
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		for (size_t j = 0; j < m; j++) {
			u_max = u_max > b[i][j] ? u_max : b[i][j];
		}
	}
	
	MPI_Allreduce(&u_max, &u_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	/*
	 * Compute the maximum pointwise error for convergence testing.
	 */
	real max_error = 0.0;
	// Work already distributed for m
	// Add openmp, accessing same max_error
	#pragma omp parallel for reduction(max : max_error)
    	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
        	for (size_t j = 0; j < m; j++) {
            		real sol = solution(grid[i+1], grid[j+1]);
            		real error = fabs(sol - b[i][j]);
            		max_error = max_error > error ? max_error : error;
        	}
    	}

	MPI_Allreduce(&max_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (rank == 0) {
		duration = MPI_Wtime() - time_start;
		printf("duration: %e\n", duration);
		printf("u_max = %e\n", u_max);
		printf("max_error = %e\n", max_error);
	}
}

/*
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to swtich between problem definitions.
 */

real rhs(real x, real y) {
	//return 2 * (y - y*y + x - x*x);
	return 5 * pow(PI, 2) * sin(PI*x) * sin(2*PI*y);
}

/*
 * Exact solution used to perform a convergence test.
 */

real solution(real x, real y) {
	return sin(PI*x) * sin(2*PI*y);
}

/*
 * Write the transpose of b a matrix of R^(m*m) in bt.
 * In parallel the function MPI_Alltoallv is used to map directly the entries
 * stored in the array to the block structure, using displacement arrays.
 */

void transpose(real **bt, real **b, size_t m, int* work, int* senddisps, int* sendcounts, int* recvdisps, int* recvcounts)
{
	int nprocs, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//if (rank == 0) { print_matrix(b, m); }
	
	// Each process allocates their assigned set of transposed rows from the original columns
	#pragma omp parallel for schedule(static)
	for (size_t i = work[0]; i < work[0] + work[1]; i++) {
		for (size_t j = 0; j < m; j++) {
			bt[i][j] = b[j][i];
		}
	}

	// Every process sends their transposed rows, and stores received ones in bt
	MPI_Alltoallv(bt[0], sendcounts, senddisps, MPI_DOUBLE, bt[0], recvcounts, recvdisps, MPI_DOUBLE, MPI_COMM_WORLD);

	
	//if (rank == 0) { print_matrix(bt, m); }
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

// Load balancing is handled on each processor as this slightly improved stability on more processors, and should not provide any added overhead given the nature of the program
void distribute_work(int total, int *work, int *recvdisps, int *recvcounts) 
{
	int nprocs, rank;
	int tag = 100;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int index = 0;
	int workload;
	int work_left = total;
	int p_left = nprocs;

	while (p_left > 0) {
		int p = nprocs-p_left;
		workload = work_left/p_left;
		
		recvdisps[p] = index;
		recvcounts[p] = workload;
		if (p == rank){
			work[0] = index;
			work[1] = workload;
		}
		
		index += workload;
		work_left -= workload;
		p_left--;
	}
}

void print_matrix(real **matrix, int m){
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < m; j++) {
			printf("%f	", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
