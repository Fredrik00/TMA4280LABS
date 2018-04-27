#include <stdio.h>
#include <math.h>
#include "zeta1.hpp"
#include <iostream>
#include <fstream>
using namespace std;

double zeta(int n) {
	const int p = 4;
	int p_left = p;
	int work_left = n;
	int index = 0;
	int indices[p];
	int work_loads[p];
	double sum = 0.0;
	double v[n];
	
	for (int i = 0; i < p; i++) {
		work_loads[i] = work_left/p_left;
		work_left -= work_loads[i];
		p_left--;
		indices[i] = index;
		index += work_loads[i];
	}

	for (int i = 0; i < p; i++) {
		printf("index %i, length %i\n", indices[i], work_loads[i]);
		zeta_worker(v, indices[i], work_loads[i]);
	}
	
	for (int i = 0; i < n; i++) {
		sum += v[i];
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
		for (int k = 1; k < 25; k++) {
			int n = pow(2, k);
			error = fabs(M_PI - zeta(n));
			file << n << "	error	" << error << "\n";
		}
		file.close();
	}
}

void zeta_worker(double v[], int index, int length){
	for (int i = index; i < index + length; i++){  // i starts at 0 rather than 1
		v[i] = 1.0/pow(i+1, 2);
	}
}
