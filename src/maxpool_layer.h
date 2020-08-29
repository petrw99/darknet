#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "dark_cuda.h"
#include "layer.h"
#include "network.h"

typedef layer maxpool_layer;

#ifdef __cplusplus
extern "C" {
#endif
image get_maxpool_image(maxpool_layer l);
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride_x, int stride_y, int padding, int maxpool_depth, int out_channels, int antialiasing, int avgpool, int train);
void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);

void forward_local_avgpool_layer(const maxpool_layer l, network_state state);
void backward_local_avgpool_layer(const maxpool_layer l, network_state state);

#ifdef GPU
<<<<<<< HEAD
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);
#endif
=======
void forward_maxpool_layer_gpu(maxpool_layer l, network_state state);
void backward_maxpool_layer_gpu(maxpool_layer l, network_state state);
void cudnn_maxpool_setup(maxpool_layer *l);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4

void forward_local_avgpool_layer_gpu(maxpool_layer layer, network_state state);
void backward_local_avgpool_layer_gpu(maxpool_layer layer, network_state state);
#endif // GPU

#ifdef __cplusplus
}
#endif

#endif
