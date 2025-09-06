#include <stdlib.h>
#include "utils.h"

float urand_sym(float bound) {
    float u = (float)rand() / ((float)RAND_MAX + 1.0f);   // [0,1)
    return (2.0f*u - 1.0f) * bound;                       // [-bound, +bound)
}
