#include "neural_network.h"
#include <stdio.h>
#include "print_mv.h"

int main(int argc, char *argv[]){
	//hadamard working with no errors
	gsl_matrix *m1 = gsl_matrix_alloc(2,3);
	gsl_matrix *m2 = gsl_matrix_alloc(2,3);

	size_t rows = m1 -> size1;
	size_t cols = m2 -> size2;

	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			gsl_matrix_set(m1, i, j, i+1+j);
			gsl_matrix_set(m2, i, j, 6-i-j);
		}
	}

	print_matrix(m1);
	print_matrix(m2);

	gsl_matrix_mul_elements(m1, m2);

	print_matrix(m1);

	gsl_matrix_free(m1);
	gsl_matrix_free(m2);
}
