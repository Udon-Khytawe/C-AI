#include "neural_network.h"
#include "print_mv.h"
#include <stdio.h>

int main(int argc, char *argv[]){
	//read in files and stuff
	FILE *nums = fopen("data/nums", "rb");
	FILE *labels = fopen("data/labels", "rb");

	char buf;
	fread(&buf, sizeof(char), 1, nums);
	int rows = (int)buf;

	fread(&buf, sizeof(char), 1, nums);
	int cols = (int)buf;

	gsl_matrix **images = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*10);
	for(int i = 0; i < 10; ++i){
		images[i] = gsl_matrix_calloc(rows*cols, 1);//use calloc because most values are not initilized
		for(int j = 0; j < rows*cols; ++j){
			fread(&buf, sizeof(char), 1, nums);
			gsl_matrix_set(images[i], j, 0, (int)buf);
		}
	}

	char lbls[10] = {0};
	fread(lbls, sizeof(char), 10, labels);
	gsl_matrix **label_mats = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*10);
	for(int i = 0; i < 10; ++i){
		label_mats[i] = gsl_matrix_calloc(10, 1);
		gsl_matrix_set(label_mats[i], (int)lbls[i], 0, 1);
	}

	fclose(nums);
	fclose(labels);

	//make the ai do ai stuff
	srand(time(NULL));
	size_t size = 3;
	size_t sizes[] = {25, 15, 10};
	func acts[] = {sigmoid, sigmoid, sigmoid};

	neural_network *net = neural_network_alloc(size, sizes);
	neural_network_init(net, acts, ssd, -1, 1);

	size_t num_train = 10000;
	gsl_matrix **test = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*num_train);
	gsl_matrix **real = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*num_train);
	for(int i = 0; i < num_train; ++i){
		int index = rand()%10;
		test[i] = images[index];
		real[i] = label_mats[index];
	}

	train_neural_network(num_train, 10, 1, test, real, net);
	int correct = 0;
	int guesses[10] = {0};
	gsl_matrix *out = gsl_matrix_alloc(10, 1);
	for(int i = 0; i < 10; ++i){
		feed_forward(out, images[i], net);

		size_t outr;
		size_t outc;
		size_t corr;
		size_t corc;
		gsl_matrix_max_index(out, &outr, &outc);
		gsl_matrix_max_index(label_mats[i], &corr, &corc);
		if(outr == corr){
			++correct;
		}
		guesses[outr]++;
	}

	printf("correct %d/10\n", correct);

	for(int i = 0; i < 10; ++i){
		gsl_matrix_free(images[i]);
		gsl_matrix_free(label_mats[i]);
	}
	free(images);
	free(label_mats);
	free(test);
	free(real);
	gsl_matrix_free(out);
	neural_network_free(net);
}
