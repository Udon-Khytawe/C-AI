#include "neural_network.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	size_t size = 5;
	size_t sizes[] = {2,100,20,10,2};
	//func acts[] = {sigmoid, sigmoid, sigmoid, sigmoid};
	func acts[] = {relu, relu, sigmoid, softmax};

	neural_network *net = neural_network_alloc(size, sizes);
	neural_network_init(net, acts, ssd, -1, 1);

	/***get training data***/
	gsl_matrix *one = gsl_matrix_alloc(2, 1);
	gsl_matrix_set(one, 0, 0, 1);
	gsl_matrix_set(one, 1, 0, 0);

	gsl_matrix *two = gsl_matrix_alloc(2, 1);
	gsl_matrix_set(two, 0, 0, 0);
	gsl_matrix_set(two, 1, 0, 1);

	size_t num_examples = 10000000;
	size_t batch_size = 5;
	gsl_matrix **inputs = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*num_examples);
	gsl_matrix **reals = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*num_examples);

	for(int i = 0; i < num_examples; ++i){
		if(rand() % 2 == 0){
			inputs[i] = one;
			reals[i] = two;
		} else {
			inputs[i] = two;
			reals[i] = one;
		}
	}
	/*********************/

	train_neural_network(num_examples, batch_size, 2, inputs, reals, net);

	gsl_matrix *out = gsl_matrix_alloc(2, 1);
	feed_forward(out, one, net);
	printf("test with 1 0:\n");
	print_matrix(out);

	feed_forward(out, two, net);
	printf("test with 0 1:\n");
	print_matrix(out);

	gsl_matrix_free(one);
	gsl_matrix_free(two);
	gsl_matrix_free(out);
	neural_network_free(net);
	free(inputs);
	free(reals);
}
