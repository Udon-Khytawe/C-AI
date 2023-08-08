#include "funcs.h"

int softmax(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v){
	//make sure the sizes are all the same
	size_t func_size1 = func_dest -> size1;
	size_t func_size2 = func_dest -> size2;
	size_t der_size1 = der_dest -> size1;
	size_t der_size2 = der_dest -> size2;
	size_t v_size1 = v -> size1;
	size_t v_size2 = v -> size2;

	if(func_size2 != 1 || der_size2 != 1 || v_size2 != 1 || func_size1 != der_size1 || func_size1 != v_size1 || der_size1 != v_size1){
		return 1;
	}

	gsl_matrix *v1 = gsl_matrix_alloc(v -> size1, v -> size2);
	gsl_matrix_memcpy(v1, v);

	size_t size = func_size1;

	double sum = 0;//sum of e^v_i
	double sum_sqrd = 0;//sum^2

	//subtract the max value from v so it doesn't break 
	double max = gsl_matrix_max(v1);
	gsl_matrix_add_constant(v1, -1*max + 1);

	//compute the sum of e^v_i
	for(int i = 0; i < size; ++i){
		sum += exp(gsl_matrix_get(v1, i, 0));
		if(isinf(sum) || isnan(sum)){//make sure sum is not infinity 
			sum = DBL_MAX;
			break;
		}
	}

	sum_sqrd = sum*sum;//get sum^2
	if(isinf(sum_sqrd) || isnan(sum_sqrd)){//make sure sum_sqrd is not infinity
		sum_sqrd = DBL_MAX;
	}

	//set the value in each of the new vectors
	for(int i = 0; i < size; ++i){
		double e_value = exp(gsl_matrix_get(v1,i,0));
		if(isnan(e_value) || isinf(e_value)){
			e_value = DBL_MAX;
		}
		gsl_matrix_set(func_dest, i, 0, e_value/sum);//set the function value
		gsl_matrix_set(der_dest, i, 0, (sum/sum_sqrd)*e_value - e_value*e_value/sum_sqrd);//set the derivitive value
	}

	gsl_matrix_free(v1);
	return 0;
}

int relu(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v){
	//make sure the sizes are all correct
	size_t func_size1 = func_dest -> size1;
	size_t func_size2 = func_dest -> size2;
	size_t der_size1 = der_dest -> size1;
	size_t der_size2 = der_dest -> size2;
	size_t v_size1 = v -> size1;
	size_t v_size2 = v -> size2;

	if(func_size2 != 1 || der_size2 != 1 || v_size2 != 1 || func_size1 != der_size1 || func_size1 != v_size1 || der_size1 != v_size1){
		return 1;
	}

	size_t size = func_size1;

	//compute the relu values
	for(int i = 0; i < size; ++i){
		double value = gsl_matrix_get(v, i, 0);

		if(isnan(value) || isinf(value)){//make sure values are not nan or inf
			gsl_matrix_set(func_dest, i, 0, DBL_MAX);
			gsl_matrix_set(der_dest, i, 0, 1);
		} else if(value > 0){
			gsl_matrix_set(func_dest, i, 0, value);
			gsl_matrix_set(der_dest, i, 0, 1);
		} else {
			gsl_matrix_set(func_dest, i, 0, 0);
			gsl_matrix_set(der_dest, i, 0, 0);
		}
	}

	return 0;
}

int leaky_relu(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v){
	//make sure the sizes are all correct
	size_t func_size1 = func_dest -> size1;
	size_t func_size2 = func_dest -> size2;
	size_t der_size1 = der_dest -> size1;
	size_t der_size2 = der_dest -> size2;
	size_t v_size1 = v -> size1;
	size_t v_size2 = v -> size2;

	if(func_size2 != 1 || der_size2 != 1 || v_size2 != 1 || func_size1 != der_size1 || func_size1 != v_size1 || der_size1 != v_size1){
		return 1;
	}

	size_t size = func_size1;

	//compute the relu values
	for(int i = 0; i < size; ++i){
		double value = gsl_matrix_get(v, i, 0);

		if(isnan(value) || isinf(value)){//make sure values are not nan or inf
			gsl_matrix_set(func_dest, i, 0, DBL_MAX);
			gsl_matrix_set(der_dest, i, 0, 1);
		} else if(value > 0){
			gsl_matrix_set(func_dest, i, 0, value);
			gsl_matrix_set(der_dest, i, 0, 1);
		} else {
			gsl_matrix_set(func_dest, i, 0, 0.01*value);
			gsl_matrix_set(der_dest, i, 0, 0.01);
		}
	}

	return 0;
}

int sigmoid(gsl_matrix *func_dest, gsl_matrix *der_dest, gsl_matrix *v){
	//make sure the sizes are all correct
	size_t func_size1 = func_dest -> size1;
	size_t func_size2 = func_dest -> size2;
	size_t der_size1 = der_dest -> size1;
	size_t der_size2 = der_dest -> size2;
	size_t v_size1 = v -> size1;
	size_t v_size2 = v -> size2;

	if(func_size2 != 1 || der_size2 != 1 || v_size2 != 1 || func_size1 != der_size1 || func_size1 != v_size1 || der_size1 != v_size1){
		return 1;
	}

	size_t size = func_size1;

	for(int i = 0; i < size; ++i){
		double value = gsl_matrix_get(v, i, 0);
		double func = 0;
		double der = 0;
		if(value > 0){
			value = exp(-1*value);
			func = 1/(1+value);
			der = value/((1+value)*(1+value));
		} else {
			value = exp(value);
			func = value/(1+value);
			der = value/((1+value)*(1+value));
		}
		gsl_matrix_set(func_dest, i, 0, func);
		gsl_matrix_set(der_dest, i, 0, der);
	}

	return 0;
}

int ssd(gsl_matrix *der_dest, gsl_matrix *output, gsl_matrix *expected){
	//make sure the sizes are all correct
	size_t der_size1 = der_dest -> size1;
	size_t der_size2 = der_dest -> size2;
	size_t output_size1 = output -> size1;
	size_t output_size2 = output -> size2;
	size_t expected_size1 = expected -> size1;
	size_t expected_size2 = expected -> size2;

	if(der_size2 != 1 || output_size2 != 1 || expected_size2 != 1 || der_size1 != output_size1 || der_size1 != expected_size1 || output_size1 != expected_size1){
		return 1;
	}

	size_t size = der_size1;

	//compute the derivitive values
	for(int i = 0; i < size; ++i){
		double value = 2*(gsl_matrix_get(output, i, 0) - gsl_matrix_get(expected, i, 0));
		if(isnan(value) || isinf(value)){
			value = DBL_MAX;
		}
		gsl_matrix_set(der_dest, i, 0, value);
	}

	return 0;
}
