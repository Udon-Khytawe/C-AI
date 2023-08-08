#include "mnist_reader.h"

labels* labels_alloc_read(FILE *stream){
	fseek(stream, 0, SEEK_SET);//get to the start of the file
	unsigned char buf[4];

	//make sure the magic number is 2049
	int32_t magic;
	fread(buf, sizeof(int32_t), 1, stream);

	magic = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
	if(magic != 2049){
		return NULL;
	}

	//get the number of labels
	int32_t len;
	fread(buf, sizeof(int32_t), 1, stream);
	len = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];

	//allocate memory for the label and the labels 
	labels* l = (labels*)malloc(sizeof(labels));
	l -> size = len;
	l -> labels = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*len);

	//copy the data to the gsl_matrices
	unsigned char sbuf;
	size_t data_range = 10;
	for(int i = 0; i < l -> size; ++i){
		l -> labels[i] = gsl_matrix_calloc(data_range, 1);
		fread(&sbuf, sizeof(unsigned char), 1, stream);
		gsl_matrix_set(l -> labels[i], (int)sbuf, 0, 1);
	}

	return l;
}

void labels_free(labels *l){
	for(int i = 0; i < l -> size; ++i){
		gsl_matrix_free(l -> labels[i]);
	}
	free(l -> labels);
	free(l);
}

images* images_alloc_read(FILE *stream){
	fseek(stream, 0, SEEK_SET);//get to the start of the file
	unsigned char buf[4];

	//make sure the magic number is 2051
	int32_t magic;
	fread(buf, sizeof(int32_t), 1, stream);

	magic = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
	if(magic != 2051){
		return NULL;
	}

	//get the number of labels
	int32_t len;
	fread(buf, sizeof(int32_t), 1, stream);
	len = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];

	//get the rows and columns
	int32_t rows;
	int32_t cols;
	fread(buf, sizeof(int32_t), 1, stream);
	rows = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
	fread(buf, sizeof(int32_t), 1, stream);
	cols = buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];


	//allocate memory for the label and the labels
	images *img = (images*)malloc(sizeof(images)*len);
	img -> size = len;
	img -> mat_size = rows*cols;
	img -> images = (gsl_matrix**)malloc(sizeof(gsl_matrix*)*len);

	//copy the data to the gsl_matricies
	unsigned char sbuf;
	for(int i = 0; i < img -> size; ++i){
		//allocate memory for matrix
		(img -> images)[i] = gsl_matrix_alloc(img -> mat_size, 1);

		//read the data from the file and the set in the matrix
		for(int j = 0; j < img -> mat_size; ++j){
			fread(&sbuf, sizeof(unsigned char), 1, stream);
			gsl_matrix_set((img -> images)[i], j, 0, (double)sbuf);
		}
	}

	return img;
}

void images_free(images* img){
	for(int i = 0; i < img -> size; ++i){
		gsl_matrix_free(img -> images[i]);
	}

	free(img -> images);
	free(img);
}
