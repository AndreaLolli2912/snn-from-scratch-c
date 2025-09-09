// net.h
#ifndef NET_H
#define NET_H

#include <stddef.h>

/* ===== Runtime layer wrapper ===== */
typedef enum { LAYER_LINEAR, LAYER_LEAKY } LayerKind;

typedef struct {
    LayerKind kind;
    void *ptr;
    int (*forward)(void *ptr, const float *in, float *out);
    int (*del)    (void *ptr);
    int (*reset)  (void *ptr);
} Layer;

/* ===== Net container ===== */
typedef struct Net {
    size_t  n_layers;
    size_t  capacity;
    Layer  *layers;

    // method pointers (match net.c)
    int (*add_linear_layer)  (struct Net *self, int in_features, int out_features, int use_bias);
    int (*add_leaky_layer)   (struct Net *self, float beta, float threshold); // <-- updated
    int (*delete_last_layer) (struct Net *self);
    void (*del_net)          (struct Net *self);
} Net;

/* ===== Public API ===== */
void init_net(Net *net);

/* (Optional) direct calls to the same functions used by the method pointers */
int add_linear_layer  (Net *net, int in_features, int out_features, int use_bias);
int add_leaky_layer   (Net *net, float beta, float threshold);  // <-- updated
int delete_last_layer (Net *net);
int net_forward       (Net *net, const float *in, float *out, size_t n_steps, size_t n_samples, size_t n_inputs);
void del_net(Net *net);

#endif // NET_H
