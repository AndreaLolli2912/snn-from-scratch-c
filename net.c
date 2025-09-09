// net.c
#include "net.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================ STRUCTURE ================ */
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

/* ================ LINEAR METHODS ================*/
static int linear_layer_forward(void *ptr, const float *in, float *out)
{
    if (!ptr || !in || !out) return 0;
    Linear *lin = (Linear*)ptr;
    const size_t in_f  = lin->in_features;
    const size_t out_f = lin->out_features;

    for (size_t o = 0; o < out_f; ++o) {
        size_t row = o * in_f;
        float acc = 0.0f;
        for (size_t i = 0; i < in_f; ++i) {
            acc += lin->weight[row + i] * in[i];
        }
        if (lin->use_bias && lin->bias) acc += lin->bias[o];
        out[o] = acc;
    }
    return 1;
}

static int linear_layer_del(void *ptr)
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
static int leaky_layer_forward (void *ptr, const float *in, float *out)
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

static int leaky_layer_del(void *ptr)
{
    if (!ptr) return 1;
    Leaky *leaky = (Leaky*)ptr;
    free(leaky->membrane);
    leaky->membrane = NULL;
    free(leaky);
    return 1;
}

static int leaky_layer_reset(void *ptr)
{
    if (!ptr) return 0;
    Leaky* leaky = (Leaky*)ptr;
    if (!leaky || !leaky->membrane) return 0;
    for (size_t k = 0; k < leaky->n; k++) {
        leaky->membrane[k] = 0.0f;
    }
    return 1;
}

/* ================ LAYER CONSTRUCTORS ================ */
static int init_linear_layer (Layer *layer, int in_features, int out_features, int use_bias)
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
    layer->reset   = NULL;
    return 1;
}

static int init_leaky_layer (Layer *layer, size_t n, float beta, float threshold)
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
    layer->reset   = leaky_layer_reset;
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

int net_forward(Net *net, const float *in, float *out, size_t n_steps, size_t n_samples, size_t n_inputs)
{
    if (!net || net->n_layers == 0 || !in || !out || n_steps == 0 || n_samples == 0 || n_inputs == 0) return 0;

    size_t maxw = n_inputs;
    size_t expect_in = n_inputs;

    for (size_t k = 0; k < net->n_layers; ++k) {
        Layer *L = &net->layers[k];
        if (L->kind == LAYER_LINEAR) {
            Linear *P = (Linear*)L->ptr;
            if ((size_t)P->in_features != expect_in) {
                fprintf(stderr, "net_forward(): layer %zu expects %zu, got %zu\n", k, (size_t)P->in_features, expect_in);
                return 0;
            }
            expect_in = (size_t)P->out_features;
            if (expect_in > maxw) maxw = expect_in;
        } else { // LAYER_LEAKY
            Leaky *Q = (Leaky*)L->ptr;
            if (Q->n != expect_in) {
                fprintf(stderr, "net_forward(): leaky %zu size %zu != prev width %zu\n", k, Q->n, expect_in);
                return 0;
            }
        }
    }
    size_t out_dim = expect_in; // final output width (last layer)

    float *bufA = malloc(maxw * sizeof *bufA);
    float *bufB = malloc(maxw * sizeof *bufB);

    if (!bufA || !bufB) {
        free(bufA);
        free(bufB);
        return 0;
    }

    for (size_t s = 0; s < n_samples; s++){
        // reset layers membrane for each sample
        for (size_t k = 0; k < net->n_layers; ++k) {
            Layer *L = &net->layers[k];
            if (L->kind == LAYER_LEAKY && L->reset) {
                L->reset(L->ptr);
            }
        }
        for (size_t t = 0; t < n_steps; t++) {
            const float *in_pin  = in  + s*(n_steps*n_inputs) + t*n_inputs;
            float *out_pin = out + s*(n_steps*out_dim) + t*out_dim;
            memcpy(bufA, in_pin, n_inputs * sizeof *bufA);

            size_t curW = n_inputs;
            for (size_t layer_idx = 0; layer_idx < net->n_layers; layer_idx++){
                Layer *L = &net->layers[layer_idx];
                if (!L->forward(L->ptr, bufA, bufB)){
                    free(bufA);
                    free(bufB);
                    return 0;
                }
                curW = (L->kind == LAYER_LINEAR)
                   ? (size_t)((Linear*)L->ptr)->out_features
                   : ((Leaky*)L->ptr)->n;
                float *tmp = bufA; bufA = bufB; bufB = tmp;
            }
            memcpy(out_pin, bufA, curW * sizeof *out_pin);
        }
    }
    free(bufA);
    free(bufB);
    return 1;
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



