#include <gsl/gsl_matrix.h>
#include <math.h>
#include <float.h>

#ifndef FUNCS_H
#define FUNCS_H

typedef int (*func)(gsl_matrix*, gsl_matrix*, gsl_matrix*);

int norm_softmax(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v);

int softmax(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v);

int relu(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v);

int leaky_relu(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v);

int sigmoid(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v);
//just used for the derivitive since the cost is not useful
int ssd(gsl_matrix *der_dest, gsl_matrix *output, gsl_matrix *expected);

#endif 
