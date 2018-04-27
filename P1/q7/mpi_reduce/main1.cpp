#include "zeta1.hpp"
#include "mach1.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>


int main(int argc, char* argv[]) {
	int n = 10000;
	int nprocs, rank;
	double time_start;
	double duration;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (rank == 0) {
		if (nprocs != 0 && (nprocs & (nprocs-1)) != 0) {
			printf("Number of processes must be a power of 2\n");
			MPI_Abort(MPI_COMM_WORLD, 0);
			return 0;
		}
		time_start = MPI_Wtime();
	}
	
	if (argc > 1) {
		if (strcmp(argv[1], "utest") == 0) {
			zutest();
			mutest();
			MPI_Finalize();
			if (rank == 0) {
				duration = MPI_Wtime() - time_start;
				printf("Total wall time:  %f\n", duration);
			}
			return 0;
		}

		else if (strcmp(argv[1], "vtest") == 0) {
			zvtest();
			mvtest();
			MPI_Finalize();
			if (rank == 0) {
				duration = MPI_Wtime() - time_start;
				printf("Total wall time:  %f\n", duration);
			}
			return 0;
		}
			
		else { n = atoi(argv[1]); }
	}
	
	double z = zeta(n);
	double m = mach(n);
	if (rank == 0) {
		printf("%f\n", z);
		printf("%f\n", m);
		duration = MPI_Wtime() - time_start;
		printf("Total wall time:  %f\n", duration);
	}
	
	MPI_Finalize();	

	return 0;
}
