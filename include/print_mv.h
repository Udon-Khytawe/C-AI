#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#ifndef PRINT_MV_H
#define PRINT_MV_H

void print_matrix(gsl_matrix *m);

void print_vector(gsl_vector *v);

#endif
