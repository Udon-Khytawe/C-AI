#include "neural_network.h"

gsl_matrix* rand_gsl_matrix_alloc(size_t rows, size_t cols, double lower, double upper){

	if(lower > upper)//if lower is greater than upper
		return NULL;//return null

	double range = upper - lower;//get the range

	gsl_matrix *m = gsl_matrix_alloc(rows, cols);//allocate memory

	//set every element to a random value
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			double random = range*(((double)rand())/RAND_MAX) + lower;
			gsl_matrix_set(m, i, j, random);
		}
	}

	return m;
}

neural_network* neural_network_alloc(size_t num_layers, size_t *layer_sizes){
	
	//allocate memory to a neural network struture
	neural_network *net = (neural_network*)malloc(sizeof(neural_network));

	net -> num_layers = num_layers;

	//allocate memory to the list of layers
	net -> layer_sizes = (size_t*)malloc(sizeof(size_t)*num_layers);

	//copy layer_sizes into the network 
	memcpy(net -> layer_sizes, layer_sizes, sizeof(size_t)*num_layers);

	//allocate memory to the activation functions
	int len = num_layers - 1;
	net -> acts = (func*)malloc(sizeof(func)*(len));

	//allocate memory to the list of weight matrices
	net -> weights = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*(len));

	//allocate memory to the list of bias vectors
	net -> biases = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*(len));

	//allocate memory to the intermidate zed vectors
	net -> activations = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*(len+1));

	//allocate memory to the intermediate activation vectors
	net -> activation_primes = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*(len));

	net -> cost = NULL;//set the cost function to null
	net -> initilized = 0;//set the initlized value to false

	return net;
}

void neural_network_init(neural_network *network, func *acts, func cost, double lower, double upper){
	//TODO have random seed generation
	//srand(time(NULL));
	//set the network to initilized
	network -> initilized = 1;

	//set the cost function
	network -> cost = cost;

	//allocate memory for weights and biases and intermediate values
	int len = network -> num_layers - 1;//wb_end is the weights and biases last index+1
	size_t *layer_sizes = network -> layer_sizes;

	for(int i = 0; i < len; ++i){
		(network -> weights)[i] = rand_gsl_matrix_alloc(layer_sizes[i+1], layer_sizes[i], lower, upper);
		(network -> biases)[i] = rand_gsl_matrix_alloc(layer_sizes[i+1], 1, lower, upper);
		(network -> activations)[i] = gsl_matrix_alloc(layer_sizes[i], 1);
		(network -> activation_primes)[i] = gsl_matrix_alloc(layer_sizes[i+1], 1);
		(network -> acts)[i] = acts[i];//copy activation functions into the network
	}

	(network -> activations)[len] = gsl_matrix_alloc(layer_sizes[len], 1);
}

void neural_network_free(neural_network *network){
	if(1 == network -> initilized){
		int weights_len = network -> num_layers - 1;
		
		//free the weights and biases and intermediates
		for (int i = 0; i < weights_len; ++i){
			gsl_matrix_free((network -> weights)[i]);
			gsl_matrix_free((network -> biases)[i]);
			gsl_matrix_free((network -> activations)[i]);
			gsl_matrix_free((network -> activation_primes)[i]);
		}

		//free the final activation and activation prime layers
		gsl_matrix_free((network -> activations)[weights_len]);
	}

	free(network -> weights);//free the weights list
	free(network -> biases);//free the biases list
	free(network -> activations);//free the activation values
	free(network -> activation_primes);//free the activation derivitive values
	free(network -> acts);//free the activation function list
	free(network -> layer_sizes);//free the layer sizes list
	free(network);//free the network itself
}

gradient* gradient_alloc(neural_network *network){
	
	//allocate memory to the gradient
	gradient *grad = (gradient*)malloc(sizeof(gradient));
	
	//set the number of layers and allocate memory for layer sizes
	grad -> num_layers = network -> num_layers;
	grad -> layer_sizes = (size_t*)malloc(sizeof(size_t)*(grad -> num_layers));
	
	size_t num_layers = grad -> num_layers;//get number of layer
	size_t *layer_sizes = grad -> layer_sizes;//get the layer sizes

	//copy layer sizes from network into gradient
	memcpy(layer_sizes, network -> layer_sizes, sizeof(size_t)*num_layers);

	//allocate memory for the weight and bias partial derivitives
	int len = grad -> num_layers - 1;//get the length
	grad -> weight_partials = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*len);
	grad -> bias_partials = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*len);

	//allocate memory for weigth and bias partial derivitives
	for(int i = 0; i < len; ++i){
		(grad -> weight_partials)[i] = gsl_matrix_calloc(layer_sizes[i+1], layer_sizes[i]);
		(grad -> bias_partials)[i] = gsl_matrix_calloc(layer_sizes[i+1], 1);
	}

	return grad;
}

void gradient_free(gradient *grad){
	
	size_t len = grad -> num_layers - 1;

	//free the weight and bias matrices
	for(int i = 0; i < len; ++i){
		gsl_matrix_free((grad -> weight_partials)[i]);
		gsl_matrix_free((grad -> bias_partials)[i]);
	}

	free(grad -> weight_partials);//free weight partial derivitves
	free(grad -> bias_partials);//free bias partial derivitives
	free(grad -> layer_sizes);//free the layer sizes
	free(grad);//free the gradient itself
}

int feed_forward(gsl_matrix *dest, gsl_matrix *input, neural_network *network){
	
	//if the input vector is the wrong size
	if(input -> size2 != 1 || input -> size1 != (network -> layer_sizes)[0]){
		return 1;
	}

	//if the output vector is the wrong size
	if(dest -> size2 != 1 || dest -> size1 != (network -> layer_sizes)[network -> num_layers - 1]){
		return 1;
	}

	size_t size = network -> num_layers - 1;
	gsl_matrix **weights = network -> weights;//get the weights matrices
	gsl_matrix **biases = network -> biases;//get the bias vectors
	gsl_matrix **activations = network -> activations;//get the activations vectors
	gsl_matrix **activation_primes = network -> activation_primes;//get the activation derivitive vectors
	func *acts = network -> acts;//get the activation functions

	//copy the input into activaitons[0]
	gsl_matrix_memcpy(activations[0], input);

	//forward propegate through the network
	for(int i = 0; i < size; ++i){
		//copy the biases into the activation layer
		gsl_matrix_memcpy(activations[i+1], biases[i]);
		
		//multiply weights by current to get 
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, weights[i], activations[i], 1, activations[i+1]);

		//apply the activation function and generate the partial derivitive 
		acts[i](activations[i+1], activation_primes[i], activations[i+1]);

	}
	
	//copy the result into the destination vector
	gsl_matrix_memcpy(dest, activations[size]);
	return 0;
}

int backpropegation(gradient *grad, gsl_matrix *input, gsl_matrix *real, neural_network *network){

	//make sure input and output layers are the correct size
	if(input -> size2 != 1 || real -> size2 != 1 || input -> size1 != network -> layer_sizes[0] || real -> size1 != network -> layer_sizes[network -> num_layers - 1]){
		return 1;
	}

	size_t size = grad -> num_layers - 1;//get the network size - 1
	size_t *sizes = network -> layer_sizes;//get network layer sizes
	gsl_matrix **weights = network -> weights;//get network weight matricies
	gsl_matrix **activations = network -> activations;//get the activations vectors
	gsl_matrix **activation_primes = network -> activation_primes;//get the activation derivitive vectors
	func cost = network -> cost;//the cost function of the network

	gsl_matrix **weight_partials = grad -> weight_partials;//get the weight partials list
	gsl_matrix **bias_partials = grad -> bias_partials;//get hte bias partials list

	gsl_matrix *output = gsl_matrix_alloc(sizes[size], 1);//allocate memory to the output
	feed_forward(output, input, network);//feedforward the network

	cost(bias_partials[size-1], output, real);
	for(int i = size - 1; i >= 0; --i){
		//get the partials for the current bias layer
		/*
		* this step is wrong when the partials are a matrix and not a vector
		* for instance with the softmax function when there is more than one element
		* that means that each layer in network -> activation_primes will need to know its size
		* :(
		*/
		gsl_matrix_mul_elements(bias_partials[i], activation_primes[i]);
		
		//get the partials for the weights
		gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, bias_partials[i], activations[i], 0, weight_partials[i]);

		//get the partials for the previoud layer storing in biases 
		if(i > 0){
			gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, weights[i], bias_partials[i], 0, bias_partials[i-1]);
		}
	}

	gsl_matrix_free(output);
	return 0;
}

void gradient_average(gradient *avrg, gradient **grads, size_t batch_size, size_t training_rate){
	//get the size of the gradients
	size_t size = avrg -> num_layers - 1;
	
	double scale = training_rate*((double)1)/batch_size;

	//iterate through each layer of the gradient
	for(int i = 0; i < size; ++i){
		//iterate through every batch
		for(int j = 0; j < batch_size; ++j){
			gsl_matrix_add((avrg -> weight_partials)[i], (grads[j] -> weight_partials)[i]);
			gsl_matrix_add((avrg -> bias_partials)[i], (grads[j] -> bias_partials)[i]);
		}

		//get the average
		gsl_matrix_scale((avrg -> weight_partials)[i], scale);
		gsl_matrix_scale((avrg -> bias_partials)[i], scale);
	}
}

int gradient_memcpy(gradient *dest, gradient *src){
	//make sure number of layers is the same
	if(dest -> num_layers != src -> num_layers){
		return 1;
	}
	size_t size = src -> num_layers - 1;
	for(int i = 0; i <= size; ++i){//make sure sizes are the same
		if(dest -> layer_sizes[i] != src -> layer_sizes[i]){
			return 1;
		}
	}

	//copy info from one gradient to the other
	for(int i = 0; i < size; ++i){
		gsl_matrix_memcpy(dest -> weight_partials[i], src -> weight_partials[i]);
		gsl_matrix_memcpy(dest -> bias_partials[i], src -> bias_partials[i]);
	}

	return 1;
}

void gradient_sub(neural_network *network, gradient *grad, double scale){

	//get the size of the network and gradient
	size_t size = network -> num_layers - 1;

	//subtract the gradient from the network 
	for(int i = 0; i < size; ++i){
		gsl_matrix_scale(grad -> weight_partials[i], scale);//scale the matricies
		gsl_matrix_scale(grad -> bias_partials[i], scale);
		gsl_matrix_sub(network -> weights[i], grad -> weight_partials[i]);//subtract the matricies
		gsl_matrix_sub(network -> biases[i], grad -> bias_partials[i]);
	}
}

int train_neural_network(size_t num_training_examples, size_t batch_size, double training_rate, double momentum, gsl_matrix **input, gsl_matrix **real, neural_network *network){

	//allocate memory for the gradients
	gradient *avrg = gradient_alloc(network);
	gradient *previous = gradient_alloc(network);//bias and weight partials are guerenteed to be zero, should probably be an calloc

	gradient **grads = (gradient**)malloc(sizeof(gradient*)*batch_size);
	for(int i = 0; i < batch_size; ++i){
		grads[i] = gradient_alloc(network);
	}

	size_t len = num_training_examples / batch_size;

	for(int i = 0; i < len; ++i){
		for(int j = 0; j < batch_size; ++j){
			//do the backpropegation getting each gradient
			backpropegation(grads[j], input[i*batch_size + j], real[i*batch_size + j], network);
		}
		//find the average gradient from the batch size
		gradient_average(avrg, grads, batch_size, training_rate);

		//subtract the average gradient from the network
		gradient_sub(network, avrg, 1);//apply the average gradient
		gradient_sub(network, previous, momentum);//apply the previoud momentum
		gradient_memcpy(previous, avrg);//copy the avrg to the previoud for momentum
	}

	//free the gradients
	for(int i = 0; i < batch_size; ++i){
		gradient_free(grads[i]);
	}

	gradient_free(avrg);
	gradient_free(previous);
	free(grads);

	return 0;
}
