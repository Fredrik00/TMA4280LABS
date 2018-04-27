#include "zeta1.hpp"
#include "mach1.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[]) {
	int n = 10000;
	
	if (argc > 1) {
		if (strcmp(argv[1], "utest") == 0) {
			zutest();
			mutest();
			return 0;
		}

		else if (strcmp(argv[1], "vtest") == 0) {
			zvtest();
			mvtest();
			return 0;
		}
			
		else { n = atoi(argv[1]); }
	}
	
	printf("%f\n", zeta(n));
	printf("%f\n", mach(n));

	return 0;
}
