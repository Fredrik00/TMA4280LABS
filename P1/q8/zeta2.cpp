#include <stdio.h>
#include <math.h>
#include "zeta2.hpp"
#include <iostream>
#include <fstream>
#include <omp.h>
using namespace std;

double zeta(int n) {
	double sum = 0.0;

	#pragma omp parallel for schedule(static) reduction(+:sum)
	for (int i = 1; i < n+1; i++) {
		sum += 1.0/pow(i, 2);
	}

	double pi = sqrt(sum*6);
	return pi;
}

void zutest() {
	int n = 3;
	double z = zeta(n);
	
	if (fabs(z - 2.857738) < 0.000001) { printf("Unit test successful\n"); }
	else { printf("Unit test error\n"); }
}

void zvtest() {
	ofstream file ("zvtest.txt");
	double error;
	if (file.is_open()) {
		double start_time;
		double duration;
		for (int k = 1; k < 25; k++) {
			int n = pow(2, k);
			start_time = omp_get_wtime();
			error = fabs(M_PI - zeta(n));
			duration = omp_get_wtime() - start_time;
			file << n << "	error	" << error << "	duration:	" << duration << "\n";
		}
		file.close();
	}
}
