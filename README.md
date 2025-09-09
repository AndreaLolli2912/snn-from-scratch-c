# SNN from scratch (C) 

Tiny C project that builds a simple spiking network and runs an inference loop:

```
Linear -> LeakyIF -> Linear -> LeakyIF
```

Inputs in `[0,1]` are converted to Bernoulli spike trains (rate coding), then unrolled over time.
The project covers part of the deploy of a Spiking Neural Network on the PULP platform.

## Layout

* `net.h/.c` – layer wrappers, Linear + LeakyIF, network builder, `net_forward`
* `spikegen.h/.c` – `spikegen_rate()` (static Bernoulli)
* `dataset.h/.c` – toy data generator in `[0,1]`
* `utils.h/.c` – RNG + indexing helpers
* `main.c` – minimal end-to-end test
* `minimal.py` – rough Python reference

## Key conventions

* **Shapes:** time-major `[T, S, N]` flattened; index with
  `idx = t*(S*N) + s*N + n`.
* **State reset:** LeakyIF membranes are reset **once per sample**.
* **Init:** Linear weights/biases adopt standard PyTorch Kaiming Uniform Initialization.
