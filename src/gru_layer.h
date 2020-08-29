
#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

<<<<<<< HEAD
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam);
=======
#ifdef __cplusplus
extern "C" {
#endif
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4

void forward_gru_layer(layer l, network state);
void backward_gru_layer(layer l, network state);
void update_gru_layer(layer l, update_args a);

#ifdef GPU
<<<<<<< HEAD
void forward_gru_layer_gpu(layer l, network state);
void backward_gru_layer_gpu(layer l, network state);
void update_gru_layer_gpu(layer l, update_args a);
=======
void forward_gru_layer_gpu(layer l, network_state state);
void backward_gru_layer_gpu(layer l, network_state state);
void update_gru_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void push_gru_layer(layer l);
void pull_gru_layer(layer l);
#endif

#ifdef __cplusplus
}
#endif

#endif
