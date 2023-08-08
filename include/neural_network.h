#include "funcs.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct {
	gsl_matrix **weights;//weight matrices
	gsl_matrix **biases;//bias vectors
	gsl_matrix **activations;//activation vectors
	gsl_matrix **activation_primes;//activation derivitive vectors
	size_t num_layers;//number of layers
	size_t *layer_sizes;//input, all hidden layers, output
	func *acts;//list of activation functions for hidden layers
	func cost;//cost function for the network
	int initilized;//whether or not the network has been initilized 1 for yes 0 otherwise
} neural_network;

typedef struct {
	gsl_matrix **weight_partials;//weight partial derivitives
	gsl_matrix **bias_partials;//bias partial derivitives
	size_t num_layers;//number of layers
	size_t *layer_sizes;//size of each layer
} gradient;

/*
 * allocates memory to a gsl_matrix and initilizes each element to a random value
 * rows is the number of rows
 * cols is the number of columns
 * lower and upper are the lower and upper bounds respectivly for the random values
 *
 * returns the random matrix
 * returns NULL if the lower bound is greater than the upper bound
 */
gsl_matrix* rand_gsl_matrix_alloc(size_t rows, size_t cols, double lower, double upper);

/*
 * allocates memory to a neual network
 * num_layers is the number of layers in the network
 * layer_sizes is an array of sizes for each layer in the network
 * 	this array is copied into the network
 * 
 * returns the initilizes neural network
 */
neural_network* neural_network_alloc(size_t num_layers, size_t *layer_sizes);

/*
 * initilizes the neural network with random wights and biases
 * this exists so I can create a new neural network
 * LATER: there should be a method for reading and writing to disk
 	hence the reason this is seperate from alloc
 * acts is an array of activation functions for each layer
 *	this array is copied into the network
 * cost is the cost function of the network
 */
void neural_network_init(neural_network *network, func *acts, func cost, double lower, double upper);

/*
 * frees the neural network
 * network is the neural network
 */
void neural_network_free(neural_network *network);

/*
 * allocates memory to a gradient based on a neural network
 * network is the neural network whose size will determine the size of the gradient memory
 */
gradient* gradient_alloc(neural_network *network);

/*
 * frees a gradient
 * grad is the gradient to be freed
 */
void gradient_free(gradient *grad);

/*
 * computes the output of the neural network for a given input
 * network is the neural network
 * input is the input vector
 * output is the output vector
 *
 * if the sizes of the input or output vector do not match with the neural networks
 * input and output sizes a 1 is returned otherwise 0.
 */
int feed_forward(gsl_matrix *dest, gsl_matrix *input, neural_network *network);

/*
 * computes the gradient for a single training example
 * network is the neural network
 * ouput is the vector output from the network
 * real is the expected output from the network with the same input
 * grad is the computed gradient for each bias and weight
 * 
 * preconditions:
 * 	grad must be the correct size for the network
 *
 * returns 1 if the input or output sizes do not match their respective sizes
 * for network, returns 0 otherwise
 */ 
int backpropegation(gradient *grad, gsl_matrix *input, gsl_matrix *real, neural_network *network);

/*
 * computes the average gradient over a given batch size
 * grads is a list of computed gradients from the batch
 * batch size is the number of gradients in the list
 * 
 * preconditions: 
 * 	all gradients must be the same size
 * 
 * returns the average gradient from the batch
 *
 * all gradients must be of the same size
 */
void gradient_average(gradient *avrg, gradient **grads, size_t batch_size, size_t training_rate);

int gradient_memcpy(gradient *dest, gradient *src);

void gradient_sub(neural_network *network, gradient *grad, double scale);

/*
 * trains the neural network on a list of inpout and output vectors'
 * network is the input neural network
 * inputs is a list of input vectors into the network
 * reals is a list of expected output vectors 
 * num_training_examples is the number of traing examples
 * batch_size is the batch size for the network
 */
int train_neural_network(size_t num_training_examples, size_t batch_size, double training_rate, double momentum, gsl_matrix **input, gsl_matrix **real, neural_network *network);

#endif
