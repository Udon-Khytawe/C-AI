#include "neural_network.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	size_t size = 3;
	size_t sizes[] = {2,3,2};
	
	neural_network *net = neural_network_alloc(size, sizes);

	gradient **grads = (gradient**)malloc(sizeof(gradient*)*2);
	grads[0] = gradient_alloc(net);
	grads[1] = gradient_alloc(net);
	gradient *avrg = gradient_alloc(net);


	for(int i = 0; i < size - 1; ++i){
		gsl_matrix_set_all(grads[0] -> weight_partials[i], 2);
		gsl_matrix_set_all(grads[0] -> bias_partials[i], 2);

		gsl_matrix_set_all(grads[1] -> weight_partials[i], 3);
		gsl_matrix_set_all(grads[1] -> bias_partials[i], 3);
	}

	gradient_average(avrg, grads, 2, 1);
	
	for(int i = 0; i < size - 1; ++i){
		print_matrix(avrg -> weight_partials[i]);
		print_matrix(avrg -> bias_partials[i]);
	}

	neural_network_free(net);
	gradient_free(grads[0]);
	gradient_free(grads[1]);
	gradient_free(avrg);
	free(grads);
}
