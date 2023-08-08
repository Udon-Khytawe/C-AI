#include "neural_network.h"
#include "mnist_reader.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	srand(time(NULL));
	size_t size = 4;
	size_t sizes[] = {784, 70, 70, 10};
	func acts[] = {sigmoid, sigmoid, sigmoid};
	
	neural_network *net = neural_network_alloc(size, sizes);
	neural_network_init(net, acts, ssd, -1.00, 1.00);
	
	printf("Network Allocated\n");
	FILE *training_images;
	FILE *training_labels;
	FILE *test_images;
	FILE *test_labels;

	training_images = fopen("data/train-images-idx3-ubyte", "rb");
	training_labels = fopen("data/train-labels-idx1-ubyte", "rb");
	test_images = fopen("data/t10k-images-idx3-ubyte", "rb");
	test_labels = fopen("data/t10k-labels-idx1-ubyte", "rb");

	printf("Files open\n");
	images *tr_imgs = images_alloc_read(training_images);
	labels *tr_lbls = labels_alloc_read(training_labels);

	images *te_imgs = images_alloc_read(test_images);
	labels *te_lbls = labels_alloc_read(test_labels);

	printf("Training Data Opened\n");
	fclose(training_images);
	fclose(training_labels);
	fclose(test_images);
	fclose(test_labels);

	printf("Training network\n");
	gsl_matrix *out = gsl_matrix_alloc(10, 1);
	for(int i = 0; i < 50; ++i){
		printf("Epoch %d\n", i+1);
		train_neural_network(tr_imgs -> size, 600, 1.00, 0.90, tr_imgs -> images, tr_lbls -> labels, net);

		int correct = 0;
		size_t guesses[10] = {0};
		for(int i = 0; i < te_imgs -> size; ++i){
			size_t outr;//out row
			size_t outc;//out col
			size_t cr;//correct row
			size_t cc;//correct col
			feed_forward(out, te_imgs -> images[i], net);
			gsl_matrix_max_index(out, &outr, &outc);
			gsl_matrix_max_index(te_lbls -> labels[i], &cr, &cc);
			//print_matrix(out);
			//print_matrix(te_lbls -> labels[i]);
			if(outr == cr){
				correct++;
			}
			guesses[outr]++;
		}

		printf("%d/%zu\n", correct, te_lbls -> size);
		for(int i = 0; i < 10; ++i){
			printf("%zu, ", guesses[i]);
		}
		printf("\n");
	}

	images_free(tr_imgs);
	images_free(te_imgs);
	labels_free(tr_lbls);
	labels_free(te_lbls);
	neural_network_free(net);
	gsl_matrix_free(out);
}
