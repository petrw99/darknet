#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "dark_cuda.h"
#include "layer.h"
#include "network.h"

<<<<<<< HEAD
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
=======
#ifdef __cplusplus
extern "C" {
#endif
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network net);
void backward_reorg_layer(const layer l, network net);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net);
void backward_reorg_layer_gpu(layer l, network net);
#endif

#ifdef __cplusplus
}
#endif

#endif
