#include "neural_network.h"
#include "print_mv.h"
#include <stdio.h>

int main(int argc, char *argv[]){
	gsl_matrix *m1 = rand_gsl_matrix_alloc(3, 3, -10, 10);
	gsl_matrix *m2 = rand_gsl_matrix_alloc(4, 3, 0, 100000);

	print_matrix(m1);
	print_matrix(m2);

	gsl_matrix *m3 = rand_gsl_matrix_alloc(3, 3, 10, -1);

	printf("isnull: %d\n", m3 == NULL);

	gsl_matrix_free(m1);
	gsl_matrix_free(m2);
	gsl_matrix_free(m3);
}
