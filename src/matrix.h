#ifndef MATRIX_H
#define MATRIX_H
#include "darknet.h"

<<<<<<< HEAD
matrix copy_matrix(matrix m);
=======
//typedef struct matrix{
//    int rows, cols;
//    float **vals;
//} matrix;

typedef struct {
    int *assignments;
    matrix centers;
} model;

#ifdef __cplusplus
extern "C" {
#endif

model do_kmeans(matrix data, int k);
matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void print_matrix(matrix m);

matrix hold_out_matrix(matrix *m, int n);
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#ifdef __cplusplus
}
#endif
#endif
