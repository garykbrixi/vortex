<div align="center">

Utilities for efficient inference of deep signal processing models (Hyena, HyenaDNA, StripedHyena2).

Standalone implementation of computational primitives for deep signal processing model architectures. For training, please refer to the [savanna](https://github.com/Zymrael/savanna/) project.

### In Docker environment

To run 40b generation sample, simply execute:

```bash
./run
```

To run 7b generation sample, simply execute:

```bash
sz=7 ./run
```

To run tests:

```bash
./run ./run_tests
```

To interactively execute commands in docker environment:

```bash
./run bash
```

### Without Docker

#### Environment setup (uv)

```bash
make setup
```

To make sure you are using the right uv environment, run `source .venv/bin/activate`

#### Run generation script

```bash
python3 generate.py \
    --config_path <PATH_TO_CONFIG> \
    --checkpoint_path <PATH_TO_CHECKPOINT> \
    --input_file <PATH_TO_INPUT_FILE> \
    --cached_generation
```

The flag `--cached_generation` is optional, but recommended for faster generation.
