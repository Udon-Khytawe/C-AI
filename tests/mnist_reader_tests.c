#include "mnist_reader.h"
#include "print_mv.h"

int main(int argc, char *argv[]){
	//labels test
	FILE *in = fopen("data/train-labels-idx1-ubyte", "rb");
	if(in == NULL){
		return 1;
	}

	labels *l = labels_alloc_read(in);

	if(l == NULL){
		fclose(in);
		return 1;
	}

	print_matrix(l -> labels[0]);
	labels_free(l);
	fclose(in);

	printf("\n");

	//images test
	in = fopen("data/t10k-images-idx3-ubyte", "rb");
	if(in == NULL){
		return 2;
	}

	images *img = images_alloc_read(in);

	gsl_matrix *seven = (img -> images)[0];
	print_matrix(seven);

	if(img == NULL){
		fclose(in);
		return 2;
	}

	images_free(img);
	fclose(in);
	return 0;
}
