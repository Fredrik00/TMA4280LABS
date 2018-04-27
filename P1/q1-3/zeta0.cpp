#include <stdio.h>
#include <math.h>
#include "zeta0.hpp"
#include <iostream>
#include <fstream>
using namespace std;

double zeta(int n) {
	double sum = 0.0;
	double vi;
	for (int i = 1; i < n+1; i++) {
		vi = 1.0/pow(i, 2);
		sum += vi;
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
