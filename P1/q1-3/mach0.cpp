#include <stdio.h>
#include <math.h>
#include "mach0.hpp"
#include <iostream>
#include <fstream>
using namespace std;

double mach(int n) {
	double arc1 = 0;
	double arc2 = 0;
	double x1 = 1.0/5.0;
	double x2 = 1.0/239.0;

	for (int i = 1; i < n+1; i++) {
		int j = 2*i-1;
		int po = pow(-1.0, i-1);
		arc1 += po*(pow(x1, j)/j);
		arc2 += po*(pow(x2, j)/j);
	}

	double sum = 4.0*arc1 - arc2;
	double pi = 4.0*sum;
	return pi;
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
