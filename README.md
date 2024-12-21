# Vortex

Utilities for efficient inference of deep signal processing models (Hyena, HyenaDNA, StripedHyena, HyenaCascade).

## Quickstart

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

Set up the environment, then run:

```bash
python3 generate.py \
    --config_path /home/zymrael/workspace/stripedhyena-2/configs/shc-evo2-7b-8k-2T-v1.yml \
    --checkpoint_path /home/zymrael/checkpoints/evo2/7b_13h_8m_8s_3a_cascade15_inference/iter_457500.pt \
    --input_file prompt.txt \
    --cached_generation
```

The flag `--cached_generation` is optional, but recommended for faster generation. 



