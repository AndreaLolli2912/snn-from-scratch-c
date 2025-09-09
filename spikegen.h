#ifndef SPIKEGEN_H
#define SPIKEGEN_H

#include <stddef.h>

void spikegen_rate(const float *in, float *out, size_t T, size_t S, size_t N);

#endif // SPIKEGEN_H
