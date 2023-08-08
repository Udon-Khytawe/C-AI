#include "neural_network.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	size_t size = 3;
	size_t layer_sizes[] = {2,3,2};
	neural_network *net = neural_network_alloc(size, layer_sizes);

	func acts[] = {relu, softmax};
	neural_network_init(net, acts, ssd, -1, 1);

	gradient *grad = gradient_alloc(net);

	gsl_matrix *input = gsl_matrix_alloc(layer_sizes[0], 1);
	gsl_matrix_set(input, 0, 0, 2);
	gsl_matrix_set(input, 1, 0, 2);

	gsl_matrix *real = gsl_matrix_alloc(layer_sizes[size-1], 1);
	gsl_matrix_set(real, 0, 0, 1);
	gsl_matrix_set(real, 1, 0, 0);

	backpropegation(grad, input, real, net);

	printf("Weights:\n");
	print_matrix((net -> weights)[0]);
	print_matrix((net -> weights)[1]);

	printf("Biases:\n");
	print_matrix((net -> biases)[0]);
	print_matrix((net -> biases)[1]);

	gsl_matrix *output = gsl_matrix_alloc(2, 1);
	feed_forward(output, input, net);

	printf("Activations:\n");
	print_matrix((net -> activations)[0]);
	print_matrix((net -> activations)[1]);
	print_matrix((net -> activations)[2]);

	printf("Activaiton derivitives:\n");
	print_matrix((net -> activation_primes)[0]);
	print_matrix((net -> activation_primes)[1]);

	printf("input and output:\n");
	print_matrix(input);
	print_matrix(output);

	printf("weight adjustments:\n");
	print_matrix((grad -> weight_partials)[0]);
	print_matrix((grad -> weight_partials)[1]);

	printf("bias adjustments:\n");
	print_matrix((grad -> bias_partials)[0]);
	print_matrix((grad -> bias_partials)[1]);

	gsl_matrix_free(output);
	gsl_matrix_free(input);
	gsl_matrix_free(real);
	neural_network_free(net);
	gradient_free(grad);
}
