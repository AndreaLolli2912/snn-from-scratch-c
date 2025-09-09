#ifndef UTILS_H
#define UTILS_H
#include <stddef.h>
float urand01(void); // in [0, 1]
float urand_sym(float bound);   // uniform in [-bound, +bound)
size_t idx_tsni(size_t t, size_t s, size_t n, size_t S, size_t N);

#endif
