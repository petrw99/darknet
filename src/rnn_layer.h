
#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#define USET

<<<<<<< HEAD
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam);
=======
#ifdef __cplusplus
extern "C" {
#endif
layer make_rnn_layer(int batch, int inputs, int hidden, int outputs, int steps, ACTIVATION activation, int batch_normalize, int log);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4

void forward_rnn_layer(layer l, network net);
void backward_rnn_layer(layer l, network net);
void update_rnn_layer(layer l, update_args a);

#ifdef GPU
<<<<<<< HEAD
void forward_rnn_layer_gpu(layer l, network net);
void backward_rnn_layer_gpu(layer l, network net);
void update_rnn_layer_gpu(layer l, update_args a);
=======
void forward_rnn_layer_gpu(layer l, network_state state);
void backward_rnn_layer_gpu(layer l, network_state state);
void update_rnn_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void push_rnn_layer(layer l);
void pull_rnn_layer(layer l);
#endif

#ifdef __cplusplus
}
#endif

#endif
