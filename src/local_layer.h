#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#include "dark_cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer local_layer;

#ifdef __cplusplus
extern "C" {
#endif
#ifdef GPU
<<<<<<< HEAD
void forward_local_layer_gpu(local_layer layer, network net);
void backward_local_layer_gpu(local_layer layer, network net);
void update_local_layer_gpu(local_layer layer, update_args a);
=======
void forward_local_layer_gpu(local_layer layer, network_state state);
void backward_local_layer_gpu(local_layer layer, network_state state);
void update_local_layer_gpu(local_layer layer, int batch, float learning_rate, float momentum, float decay, float loss_scale);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4

void push_local_layer(local_layer layer);
void pull_local_layer(local_layer layer);
#endif

local_layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation);

void forward_local_layer(const local_layer layer, network net);
void backward_local_layer(local_layer layer, network net);
void update_local_layer(local_layer layer, update_args a);

void bias_output(float *output, float *biases, int batch, int n, int size);
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size);

#ifdef __cplusplus
}
#endif

#endif
