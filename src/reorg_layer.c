#include "reorg_layer.h"
#include "dark_cuda.h"
#include "blas.h"
<<<<<<< HEAD

=======
#include "utils.h"
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
#include <stdio.h>


layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = REORG;
    l.batch = batch;
    l.stride = stride;
    l.extra = extra;
    l.h = h;
    l.w = w;
    l.c = c;
    l.flatten = flatten;
    if(reverse){
        l.out_w = w*stride;
        l.out_h = h*stride;
        l.out_c = c/(stride*stride);
    }else{
        l.out_w = w/stride;
        l.out_h = h/stride;
        l.out_c = c*(stride*stride);
    }
    l.reverse = reverse;
<<<<<<< HEAD

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    if(l.extra){
        l.out_w = l.out_h = l.out_c = 0;
        l.outputs = l.inputs + l.extra;
    }

    if(extra){
        fprintf(stderr, "reorg              %4d   ->  %4d\n",  l.inputs, l.outputs);
    } else {
        fprintf(stderr, "reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    }
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
=======
    fprintf(stderr, "reorg                    /%2d %4d x%4d x%4d -> %4d x%4d x%4d\n",  stride, w, h, c, l.out_w, l.out_h, l.out_c);
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.output = (float*)xcalloc(output_size, sizeof(float));
    l.delta = (float*)xcalloc(output_size, sizeof(float));
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4

    l.forward = forward_reorg_layer;
    l.backward = backward_reorg_layer;
#ifdef GPU
    l.forward_gpu = forward_reorg_layer_gpu;
    l.backward_gpu = backward_reorg_layer_gpu;

    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
#endif
    return l;
}

void resize_reorg_layer(layer *l, int w, int h)
{
    int stride = l->stride;
    int c = l->c;

    l->h = h;
    l->w = w;

    if(l->reverse){
        l->out_w = w*stride;
        l->out_h = h*stride;
        l->out_c = c/(stride*stride);
    }else{
        l->out_w = w/stride;
        l->out_h = h/stride;
        l->out_c = c*(stride*stride);
    }

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->outputs;
    int output_size = l->outputs * l->batch;

    l->output = (float*)xrealloc(l->output, output_size * sizeof(float));
    l->delta = (float*)xrealloc(l->delta, output_size * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void forward_reorg_layer(const layer l, network net)
{
<<<<<<< HEAD
    int i;
    if(l.flatten){
        memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            flatten(l.output, l.w*l.h, l.c, l.batch, 0);
        }else{
            flatten(l.output, l.w*l.h, l.c, l.batch, 1);
        }
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, net.input + i*l.inputs, 1, l.output + i*l.outputs, 1);
        }
    } else if (l.reverse){
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.output);
    } else {
        reorg_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 0, l.output);
=======
    if (l.reverse) {
        reorg_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, l.output);
    }
    else {
        reorg_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
    }
}

void backward_reorg_layer(const layer l, network net)
{
<<<<<<< HEAD
    int i;
    if(l.flatten){
        memcpy(net.delta, l.delta, l.outputs*l.batch*sizeof(float));
        if(l.reverse){
            flatten(net.delta, l.w*l.h, l.c, l.batch, 1);
        }else{
            flatten(net.delta, l.w*l.h, l.c, l.batch, 0);
        }
    } else if(l.reverse){
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta);
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_cpu(l.inputs, l.delta + i*l.outputs, 1, net.delta + i*l.inputs, 1);
        }
    }else{
        reorg_cpu(l.delta, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta);
=======
    if (l.reverse) {
        reorg_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, state.delta);
    }
    else {
        reorg_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, state.delta);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
    }
}

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net)
{
<<<<<<< HEAD
    int i;
    if(l.flatten){
        if(l.reverse){
            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 0, l.output_gpu);
        }else{
            flatten_gpu(net.input_gpu, l.w*l.h, l.c, l.batch, 1, l.output_gpu);
        }
    } else if (l.extra) {
        for(i = 0; i < l.batch; ++i){
            copy_gpu(l.inputs, net.input_gpu + i*l.inputs, 1, l.output_gpu + i*l.outputs, 1);
        }
    } else if (l.reverse) {
        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.output_gpu);
    }else {
        reorg_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.output_gpu);
=======
    if (l.reverse) {
        reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, l.output_gpu);
    }
    else {
        reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output_gpu);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
    }
}

void backward_reorg_layer_gpu(layer l, network net)
{
<<<<<<< HEAD
    if(l.flatten){
        if(l.reverse){
            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 1, net.delta_gpu);
        }else{
            flatten_gpu(l.delta_gpu, l.w*l.h, l.c, l.batch, 0, net.delta_gpu);
        }
    } else if (l.extra) {
        int i;
        for(i = 0; i < l.batch; ++i){
            copy_gpu(l.inputs, l.delta_gpu + i*l.outputs, 1, net.delta_gpu + i*l.inputs, 1);
        }
    } else if(l.reverse){
        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, net.delta_gpu);
    } else {
        reorg_gpu(l.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, net.delta_gpu);
=======
    if (l.reverse) {
        reorg_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, state.delta);
    }
    else {
        reorg_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, state.delta);
>>>>>>> 05dee78fa3c41d92eb322d8d57fb065ddebc00b4
    }
}
#endif
