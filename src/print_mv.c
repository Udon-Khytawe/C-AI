#include "print_mv.h"

void print_matrix(gsl_matrix *m){
	size_t rows = m -> size1;
	size_t cols = m -> size2;

	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			printf("%lf, ", gsl_matrix_get(m, i, j));
		}
		printf("\n");
	}

	printf("\n");
}

void print_vector(gsl_vector *v){
	size_t size = v -> size;

	for(int i = 0; i < size; ++i){
		printf("%lf, ", gsl_vector_get(v, i));
	}

	printf("\n\n");
}

