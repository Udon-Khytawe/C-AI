#include <stdio.h>
#include <stdint.h>
#include <gsl/gsl_matrix.h>

#ifndef MNIST_READER_H
#define MNIST_READER_H

typedef struct{
	gsl_matrix **labels;
	size_t size;
} labels;

typedef struct{
	gsl_matrix **images;//only one column
	size_t size;
	size_t mat_size;
} images;

labels* labels_alloc_read(FILE *stream);

void labels_free(labels* l);

images* images_alloc_read(FILE *stream);

void images_free(images* img);

#endif
