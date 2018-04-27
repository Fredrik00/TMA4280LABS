#include <stdio.h>
#include <math.h>
#include "mach1.hpp"
#include <iostream>
#include <fstream>
using namespace std;

double mach(int n) {
	const int p = 4;
	int p_left = p;
	int work_left = n;
	int index = 0;
	int indices[p];
	int work_loads[p];
	double sum = 0.0;
	double v[n];
	double x1 = 1.0/5.0;
	double x2 = 1.0/239.0;
	
	for (int i = 0; i < p; i++) {
		work_loads[i] = work_left/p_left;
		work_left -= work_loads[i];
		p_left--;
		indices[i] = index;
		index += work_loads[i];
	}

	for (int i = 0; i < p; i++) {
		printf("index %i, length %i\n", indices[i], work_loads[i]);
		mach_worker(v, indices[i], work_loads[i], x1, x2);
	}
	
	for (int i = 0; i < n; i++) {
		sum += v[i];
	}

	double pi = 4.0*sum;

	return pi;
}

void mach_worker(double v[], int index, int length, double x1, double x2){
	for (int i = index; i < index + length; i++){  // i starts at 0 rather than 1
		int j = 2*(i+1)-1;
		int po = pow(-1, i);
		double arc1 = po*(pow(x1, j)/j);
		double arc2 = po*(pow(x2, j)/j);
		v[i] = 4.0*arc1 - arc2;
	}
}

void mutest() {
	int n = 3;
	double m = mach(n);
	
	if (fabs(m - 3.141621) < 0.000001) { printf("Unit test successful\n"); }
	else { printf("Unit test error\n"); }
}

void mvtest() {
	ofstream file ("mvtest.txt");
	double error;
	if (file.is_open()) {
		for (int k = 1; k < 25; k++) {
			int n = pow(2, k);
			error = fabs(M_PI - mach(n));
			file << n << "	error	" << error << "\n";
		}
		file.close();
	}
}
