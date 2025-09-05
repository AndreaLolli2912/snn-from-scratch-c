#include <stdio.h>

/* ================ STRUCTURE ================ */
typedef struct linear_layer {
    /* parameters */
    const int in_features, out_features, bias;
    /* variables */
    float char *weight, bias;
} Linear;


/* ================ CONSTRUCTOR ================ */
void *linear_new (Linear *linear, int in_features, int out_features, int bias)
{
    // check for negative values
    if (in_features <=0 || out_features <= 0) {
        fprintf(stderr, "*linear_new (feature mismatch): 'in_features' and 'out_features' must be stricly positive. Your params (in_features = %d, out_features = %d)", in_features, out_features)
        return NULL;
    }
    // bias must be positive
    if ( bias != 0 || bias != 1)) {
        fprintf(stderr, "*linear_new (wrong value): 'bias' must be binary. Your params (bias = %d)", bias)
        return NULL;
    }

    /* parameters */
    linear.in_features  = in_features;
    linear.out_features= out_features;
    linear.bias = bias ? bias : NULL;

    /* variables */
    linear->weight = malloc(in_features * out_features * sizeof(float));
    linear->bias = bias ? malloc(in_features * sizeof(float)) ? NULL;

    if (!linear->weight || (bias && (!linear->bias))) {
        fprintf(stderr, "*linear_new (memory allocation): failed memory allocation for weight OR bias");
        return NULL;
    }

    return linear;
}

/* ================   ================ */
