#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam);

<<<<<<< HEAD
void forward_connected_layer(layer l, network net);
void backward_connected_layer(layer l, network net);
void update_connected_layer(layer l, update_args a);

#ifdef GPU
void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);
=======
#ifdef __cplusplus
extern "C" {
#endif
connected_layer make_connected_layer(int batch, int steps, int inputs, int outputs, ACTIVATION activation, int batch_normalize);
size_t get_connected_workspace_size(layer l);

void forward_connected_layer(connected_layer layer, network_state state);
void backward_connected_layer(connected_layer layer, network_state state);
void update_connected_layer(connected_layer layer, int batch, float learning_rate, float momentum, float decay);
void denormalize_connected_layer(layer l);
void statistics_connected_layer(layer l);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer layer, network_state state);
void backward_connected_layer_gpu(connected_layer layer, network_state state);
void update_connected_layer_gpu(connected_layer layer, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_connected_layer(connected_layer layer);
void pull_connected_layer(connected_layer layer);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
#endif

#ifdef __cplusplus
}
#endif

#endif
