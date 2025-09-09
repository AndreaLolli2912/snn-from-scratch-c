#include "utils.h"
#include <stdlib.h>
#include <stddef.h>


float urand_sym(float bound) {
    float u = (float)rand() / ((float)RAND_MAX + 1.0f);   // [0,1)
    return (2.0f*u - 1.0f) * bound;                       // [-bound, +bound)
}

float urand01(void) {
    return (float)rand() / ((float)RAND_MAX + 1.0f); // uniform in [0,1)
}

size_t idx_tsni(size_t t, size_t s, size_t n, size_t S, size_t N) {
    return t*(S*N) + s*N + n;
}
