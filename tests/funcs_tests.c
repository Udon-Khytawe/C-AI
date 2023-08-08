#include "funcs.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	size_t size = 9;
	gsl_matrix *test = gsl_matrix_alloc(size, 1);
	gsl_matrix *test_expected = gsl_matrix_alloc(size, 1);

	for(int i = 0; i < size; ++i){
		gsl_matrix_set(test, i, 0, i - 4);
		gsl_matrix_set(test_expected, i, 0, 4 - i);
	}

	printf("Test Vector: \n");
	print_matrix(test);

	printf("Test Expected Vector for cost funtions:\n");
	print_matrix(test_expected);

	gsl_matrix *func = gsl_matrix_alloc(size, 1);
	gsl_matrix *der = gsl_matrix_alloc(size, 1);

	softmax(func, der, test);
	printf("Softmax function followed by softmax derivitive:\n");
	print_matrix(func);
	print_matrix(der);

	relu(func, der, test);
	printf("Relu followed by its derivitve:\n");
	print_matrix(func);
	print_matrix(der);

	leaky_relu(func, der, test);
	printf("Leaky Relu followed by its derivitve:\n");
	print_matrix(func);
	print_matrix(der);

	ssd(der, test, test_expected);
	printf("ssd derivitve:\n");
	print_matrix(der);

	sigmoid(func, der, test);
	printf("sigmoid function because of reasons:\n");
	print_matrix(func);
	print_matrix(der);


	gsl_matrix_free(test);
	gsl_matrix_free(test_expected);
	gsl_matrix_free(func);
	gsl_matrix_free(der);
}
