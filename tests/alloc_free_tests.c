#include "neural_network.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	size_t layer_sizes[] = {2, 3, 2};
	neural_network *net = neural_network_alloc(3, layer_sizes);
	
	func acts[] = {relu, softmax};
	neural_network_init(net, acts, ssd, -10, 10);

	int size_len = net -> num_layers;
	for(int i = 0; i < size_len; ++i){
		printf("%d, ", (net -> layer_sizes)[i]);
	}
	printf("\n");
	
	print_matrix((net -> weights)[0]);
	print_matrix((net -> weights)[1]);
	print_matrix((net -> biases)[0]);
	print_matrix((net -> biases)[1]);

	gradient *grad = gradient_alloc(net);
	gradient_free(grad);

	neural_network_free(net);
}
