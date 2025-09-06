// net.h
#ifndef NET_H
#define NET_H

#include <stddef.h>  // size_t


/* ===== Runtime layer wrapper ===== */
typedef enum { LAYER_LINEAR, LAYER_LEAKY } LayerKind;

typedef struct {
    LayerKind kind;                                           // which concrete type
    void *ptr;                                                // concrete object (e.g., Linear*)
    int (*forward)(void *ptr, const float *in, float *out);   // method for this object
    int (*del)    (void *ptr);                                // destructor for this object
} Layer;

/* ===== Net container ===== */
typedef struct Net {
    size_t  n_layers;     // number of used entries
    size_t  capacity;     // allocated entries
    Layer  *layers;       // dynamic array of Layer

    // method pointers (convenience)
    int (*add_linear_layer)  (struct Net *self, int in_features, int out_features, int use_bias);
    int (*add_leaky_layer)   (struct Net *self);     // currently a stub in net.c
    int (*delete_last_layer) (struct Net *self);
} Net;

/* ===== Public API ===== */
void init_net(Net *net);

/* (Optional) direct calls to the same functions used by the method pointers */
int add_linear_layer  (Net *net, int in_features, int out_features, int use_bias);
int add_leaky_layer   (Net *net);          // stub returns 0 in net.c
int delete_last_layer (Net *net);

#endif // NET_H
