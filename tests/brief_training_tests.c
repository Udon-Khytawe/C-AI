#include "neural_network.h"
#include "print_mv.h"
void sub(neural_network *net, gradient *grad){
	size_t size = net -> num_layers - 1;

	for(int i = 0; i < size; ++i){
		gsl_matrix_sub(net -> weights[i], grad -> weight_partials[i]);
		gsl_matrix_sub(net -> biases[i], grad -> bias_partials[i]);
	}
}

int main(int argc, char *argv[]){
	size_t size = 3;
	size_t sizes[] = {2,3,2};
	func acts[] = {relu, softmax};

	neural_network *net = neural_network_alloc(size, sizes);
	neural_network_init(net, acts, ssd, -1, 1);

	gsl_matrix *one = gsl_matrix_alloc(2, 1);
	gsl_matrix_set(one, 0, 0, 1);
	gsl_matrix_set(one, 1, 0, 0);

	gsl_matrix *two = gsl_matrix_alloc(2, 1);
	gsl_matrix_set(two, 0, 0, 0);
	gsl_matrix_set(two, 1, 0, 1);

	gsl_matrix *out = gsl_matrix_alloc(2, 1);

	gradient **grads = (gradient**)malloc(sizeof(gradient*)*2);
	grads[0] = gradient_alloc(net);
	grads[1] = gradient_alloc(net);

	gradient *avrg = gradient_alloc(net);
	for(int i = 0; i < 20; ++i){
		printf("Iteration %d:\n", i);
		feed_forward(out, one, net);
		printf("1 0 output\n");
		print_matrix(out);
		feed_forward(out, two, net);
		printf("0 1 output\n");
		print_matrix(out);

		backpropegation(grads[0], one, two, net);
		backpropegation(grads[1], two, one, net);
		gradient_sub(net, grads[0]);
		gradient_sub(net, grads[1]);
	}

	neural_network_free(net);
	gradient_free(grads[0]);
	gradient_free(grads[1]);
	free(grads);
	gradient_free(avrg);
	gsl_matrix_free(one);
	gsl_matrix_free(two);
	gsl_matrix_free(out);
}
