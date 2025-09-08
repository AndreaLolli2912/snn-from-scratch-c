#ifndef SPIKEGEN_H
#define SPIKEGEN_H

static inline size_t idx_tsni(size_t t, size_t s, size_t n, size_t S, size_t N);
void spikegen_rate(const float *in, float *out, size_t T, size_t S, size_t N);

#endif // SPIKEGEN_H
