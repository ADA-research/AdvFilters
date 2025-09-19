# Filter-based Adversarial Examples for Audio Models
This repo contains the code corresponding to our paper "Robustness of Audio Classification Models against Filter Perturbations", currently under review for ICASSP 2026.  

Our code heavily relies on the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework. It is recommended to familiarise yourself with Lightning before experimenting with, or making modifications to this code.

## Basic Usage
Create a python 3.11 environment (e.g., virtualenv or conda) by using the `requirements.txt` file.

The `training` directory contains config files for all datasets and PaSST/CNN14 models. The settings given in the config files (all pgd params set to 0) simply trains the baseline models without adversarial training.

You can use them with the CLI provided in `run.py` as follows:  
```bash
python -m run fit --config './training/base_config_esc50.yaml'
```

```bash
python -m run test --config './training/base_config_esc50.yaml' --ckpt-path your_model_checkpoint.ckpt
``` 

Passing the PGD parameters `pgd_eps` (epsilon), `pgd_steps`, and `pgd_restarts` to the `fit` routine ([see documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html#train-a-model-with-the-cli)) will run adversarial training. Passing them to the `test` routine will run adversarial attacks on the test set.

Get more usage information with `python -m run -h`.

The `plotting` directory contains a jupyter notebook, as well as our results as pickle files to recreate the figures in our paper, as well as the statistical tests.

## Examples
To run adversarial training for PaSST on ESC-50 with 10 steps, 10 restarts, and epsilon = 0.5:

```bash
python -m run fit --config './training/base_config_esc50.yaml' \
 --pgd_eps 0.5 --pgd_steps 10 --pgd_restarts 10
```

## TODO
- Update requirements.txt
- model docstrings