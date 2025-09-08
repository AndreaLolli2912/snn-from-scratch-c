// net.c
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ================ STRUCTURE ================ */
typedef enum { LAYER_LINEAR, LAYER_LEAKY } LayerKind;

typedef struct {
    LayerKind kind;
    void *ptr;
    int (*forward) (void *ptr, const float *in, float *out);
    int (*del)     (void *ptr);
} Layer;

typedef struct {
    int in_features, out_features, use_bias;
    float *weight;
    float *bias;
} Linear;

typedef struct {
    size_t n;
    float beta, threshold;
    float *membrane;
} Leaky;

typedef struct Net {
    size_t n_layers;
    size_t capacity;
    Layer *layers;

    int (*add_linear_layer)  (struct Net *self, int in_features, int out_features, int use_bias);
    int (*add_leaky_layer)   (struct Net *self, float beta, float threshold);
    int (*delete_last_layer) (struct Net *self);
    void (*del_net)          (struct Net *self);
} Net;


/* ================ LINEAR METHODS ================*/
int linear_layer_forward(void *ptr, const float *in, float *out)
{
    if (!ptr || !in || !out) return 0;
    Linear *lin = (Linear*)ptr;
    const int in_f  = lin->in_features;
    const int out_f = lin->out_features;

    for (int o = 0; o < out_f; ++o) {
        float acc = 0.0f;
        const size_t row = (size_t)o * (size_t)in_f;
        for (int i = 0; i < in_f; ++i) {
            acc += lin->weight[row + (size_t)i] * in[i];
        }
        if (lin->use_bias && lin->bias) acc += lin->bias[o];
        out[o] = acc;
    }
    return 1;
}

int linear_layer_del(void *ptr)
{
    if (!ptr) return 1;
    Linear *linear = (Linear*)ptr;
    free(linear->weight);
    if (linear->bias) free(linear->bias);
    linear->weight = NULL;
    linear->bias   = NULL;
    free(linear);
    return 1;
}

/* ================ LEAKY METHODS ================*/
int leaky_layer_forward (void *ptr, const float *in, float *out)
{
    if (!ptr || !in || !out) return 0;
    Leaky *leaky = (Leaky*)ptr;

    const size_t n = leaky->n;              // also fix type (see #4)
    for (size_t i = 0; i < n; ++i) {
        float v_tmp = leaky->beta * leaky->membrane[i] + in[i];
        int spk = (v_tmp > leaky->threshold);
        leaky->membrane[i] = v_tmp - (spk ? leaky->threshold : 0.0f);
        out[i] = (float)spk;
    }
    return 1;
}

int leaky_layer_del(void *ptr)
{
    if (!ptr) return 1;
    Leaky *leaky = (Leaky*)ptr;
    free(leaky->membrane);
    leaky->membrane = NULL;
    free(leaky);
    return 1;
}

/* ================ LAYER CONSTRUCTORS ================ */
int init_linear_layer (Layer *layer, int in_features, int out_features, int use_bias)
{
    // verify arguments
    if (!layer) {
        fprintf(stderr, "init_linear_layer(): 'layer' is NULL\n");
        return 0;
    }

    if (in_features <= 0 || out_features <= 0) {
        fprintf(stderr, "init_linear_layer(): in/out must be > 0 (got %d, %d)\n",
                in_features, out_features);
        return 0;
    }
    if (use_bias != 0 && use_bias != 1) {
        fprintf(stderr, "init_linear_layer(): use_bias must be 0 or 1 (got %d)\n",
                use_bias);
        return 0;
    }

    Linear *linear = (Linear*)malloc(sizeof(*linear));
    if (!linear) {
        fprintf(stderr, "init_linear_layer(): malloc Linear failed\n");
        return 0;
    }

    // arguments
    linear->in_features  = in_features;
    linear->out_features = out_features;
    linear->use_bias     = use_bias;

    // variables
    linear->weight = (float*)malloc((size_t)out_features * (size_t)in_features * sizeof(*linear->weight));
    linear->bias   = use_bias ? (float*)malloc((size_t)out_features * sizeof(*linear->bias)) : NULL;

    if (!linear->weight || (use_bias && !linear->bias)) {
        fprintf(stderr, "init_linear_layer(): allocation failed (weight or bias)\n");
        free(linear->weight);
        free(linear->bias);
        free(linear);
        return 0;
    }

    // weight & bias initialization
    const float bound = 1.0f / sqrtf((float)linear->in_features);
    const size_t wcount = (size_t)out_features * (size_t)in_features;

    for (size_t k = 0; k < wcount; k++)
        linear->weight[k] = urand_sym(bound);

    if (use_bias) {
        for (int o = 0; o < out_features; o++)
            linear->bias[o] = urand_sym(bound);
    }

    // methods
    layer->kind = LAYER_LINEAR;
    layer->ptr = linear;
    layer->forward = linear_layer_forward;
    layer->del     = linear_layer_del;
    return 1;
}

int init_leaky_layer (Layer *layer, size_t n, float beta, float threshold)
{
    if (!layer) {
        fprintf(stderr, "init_leaky_layer(): 'layer' is NULL\n");
        return 0;
    }

    Leaky *leaky = (Leaky*)malloc(sizeof(*leaky));

    if (!leaky) {
        fprintf(stderr, "init_leaky_layer(): malloc Leaky failed\n");
        return 0;
    }
    if (n == 0) {
        free(leaky);
        return 0;
    }

    leaky->n = n;

    if (!(threshold > 0.0f)) {
        fprintf(stderr, "warning init_leaky_layer(): 'threshold' <= 0, set to 1.0f\n");
        threshold = 1.0f;
    }
    leaky->threshold = threshold;


    if (!(beta>0.0f && beta <1.0f)) {
        fprintf(stderr, "warning init_leaky_layer(): 'beta' not in (0,1), set to 0.5f\n");
        beta = 0.5f;
    }
    leaky->beta = beta;

    leaky->membrane = malloc((size_t)n * sizeof *leaky->membrane);
    if (!leaky->membrane) {
        free(leaky);
        return 0;
    }
    for (size_t i = 0; i < n; ++i) leaky->membrane[i] = 0.0f;

    layer->kind = LAYER_LEAKY;
    layer->ptr = leaky;
    layer->forward = leaky_layer_forward;
    layer->del     = leaky_layer_del;
    return 1;
}

/* ================ NET RPIVATE METHODS ================ */
static int net_reserve(Net *net, size_t cap)
{
    // verify arguments
    if (net->capacity >= cap) return 1;

    Layer *temp = realloc(net->layers, cap * sizeof(*temp));
    if (!temp) return 0;
    net->layers   = temp;
    net->capacity = cap;
    return 1;
}

/* ================ NET PUCLIC METHODS ================ */
int add_linear_layer(Net *net, int in_features, int out_features, int use_bias)
{
    // verify arguments
    if (!net) {
        fprintf(stderr, "add_linear_layer(): 'net' is NULL\n");
        return 0;
    }
    // handle network layers capacity
    if (net->n_layers == net->capacity) {
        size_t newcap = net->capacity ? net->capacity * 2 : 4;
        if (!net_reserve(net, newcap)) return 0;
    }

    Layer *slot = &net->layers[net->n_layers];
    if (!init_linear_layer(slot, in_features, out_features, use_bias)) {
        fprintf(stderr, "add_linear_layer(): init failed\n");
        return 0;
    }
    net->n_layers++;
    return 1;
}

int add_leaky_layer(Net *net, float beta, float threshold)
{
    // verify arguments
    if (!net) {
        fprintf(stderr, "add_leaky_layer(): 'net' is NULL\n");
        return 0;
    }
    if (net->n_layers == 0) {
        fprintf(stderr, "add_leaky_layer(): no previous layer to infer 'n'\n");
        return 0;
    }
    Layer *prev = &net->layers[net->n_layers - 1];
    if (prev->kind != LAYER_LINEAR) {
        fprintf(stderr, "add_leaky_layer(): previous layer is not LINEAR\n");
        return 0;
    }
    size_t n = (size_t)((Linear*)prev->ptr)->out_features;

    // handle network layers capacity
    if (net->n_layers == net->capacity) {
        size_t newcap = net->capacity ? net->capacity * 2 : 4;
        if (!net_reserve(net, newcap)) return 0;
    }

    Layer *slot = &net->layers[net->n_layers];
    if (!init_leaky_layer(slot, n, beta, threshold)) {
        fprintf(stderr, "add_leaky_layer(): init failed\n");
        return 0;
    }
    net->n_layers++;
    return 1;
}

int delete_last_layer (Net *net)
{
    if (!net || net->n_layers == 0) return 0;
    Layer *layer = &net->layers[net->n_layers - 1];
    if (layer->del) layer->del(layer->ptr);
    net->n_layers--;
    return 1;
}

void net_forward(Net *net, const float *in, float *out, int time_steps)
{
}

/* ================ NET DECONSTRUCTOR ================ */
void del_net (Net *net)
{
    while (net->delete_last_layer(net));
    free(net->layers);
    net->layers = NULL;
}

/* ================ NET CONSTRUCTOR ================ */
void init_net (Net *net)
{
    /* attributes */
    net->layers = NULL;
    net->capacity = 0;
    net->n_layers = 0;

    /* methods */
    net->add_linear_layer  = add_linear_layer;
    net->add_leaky_layer   = add_leaky_layer;
    net->delete_last_layer = delete_last_layer;
    net->del_net = del_net;
}



