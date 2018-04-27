#include "zeta1.hpp"
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
using namespace std;

double zeta(int n) {
	double sum = 0.0;
	double pi;
	int nprocs, rank;
	int work[2] = {1, 0};  // index, work load

	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	MPI_Status status;
	int tag = 100;
	if (rank ==  0) {
		int work_left = n;
		int p_left = nprocs;
	
		while (p_left > 0) {
			work[1] = work_left/p_left;
			MPI_Send(&work, 2, MPI_INT, nprocs-p_left, tag, MPI_COMM_WORLD);
			work[0] += work[1];
			work_left -= work[1];
			p_left--;
		}
	}
	
	MPI_Recv(&work, 2, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
	
	sum = zeta_worker(work[0], work[1]);

	MPI_Allreduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	pi = sqrt(pi*6);
	printf("process %i, work load %i, pi %f\n", rank, work[1], pi);

	return pi;
}

double zeta_worker(int index, int length){
	double sum = 0;
	for (int i = index; i < index + length; i++){
		sum += 1.0/pow(i, 2);
	}
	return sum;
}

void zutest() {
	int n = 3;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double z = zeta(n);
	
	if (rank == 0) {
		if (fabs(z - 2.857738) < 0.000001) { printf("Unit test successful\n"); }
		else { printf("Unit test error\n"); }
	}
}

void zvtest() {
	double error;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	ofstream file ("zvtest.txt");
	if (file.is_open()) {
		double start_time;
		double duration;
		for (int k=1; k < 25; k++) {
			int n = pow(2, k);
			if (rank == 0) { start_time = MPI_Wtime(); }
			error = fabs(M_PI - zeta(n));
			if (rank == 0) {
				duration = MPI_Wtime() - start_time;
				file << n << "	error: " << error << "	duration: " << duration << "\n";
			}
		}
		file.close();
	}
}
