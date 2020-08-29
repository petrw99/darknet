#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
<<<<<<< HEAD
void resize_normalization_layer(layer *layer, int h, int w);
void forward_normalization_layer(const layer layer, network net);
void backward_normalization_layer(const layer layer, network net);
=======
void resize_normalization_layer(layer *layer, int w, int h);
void forward_normalization_layer(const layer layer, network_state state);
void backward_normalization_layer(const layer layer, network_state state);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void visualize_normalization_layer(layer layer, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net);
void backward_normalization_layer_gpu(const layer layer, network net);
#endif

#ifdef __cplusplus
}
#endif
#endif
