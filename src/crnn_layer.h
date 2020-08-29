
#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int xnor, int train);
void resize_crnn_layer(layer *l, int w, int h);
void free_state_crnn(layer l);

void forward_crnn_layer(layer l, network net);
void backward_crnn_layer(layer l, network net);
void update_crnn_layer(layer l, update_args a);

#ifdef GPU
<<<<<<< HEAD
void forward_crnn_layer_gpu(layer l, network net);
void backward_crnn_layer_gpu(layer l, network net);
void update_crnn_layer_gpu(layer l, update_args a);
=======
void forward_crnn_layer_gpu(layer l, network_state state);
void backward_crnn_layer_gpu(layer l, network_state state);
void update_crnn_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void push_crnn_layer(layer l);
void pull_crnn_layer(layer l);
#endif

#ifdef __cplusplus
}
#endif

#endif
