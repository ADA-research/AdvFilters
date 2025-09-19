# Filter-based Adversarial Examples for Audio Models
This repo contains the code corresponding to our paper "Robustness of Audio Classification Models against Filter Perturbations", currently under review for ICASSP 2026.  

Our code heavily relies on the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework. It is recommended to familiarise yourself with Lightning before experimenting with, or making modifications to this code.

## Installation
Create a python 3.11 environment (e.g., virtualenv) and install the required modules: `pip install -r requirements.txt`

### Pre-trained Models
The pre-trained PaSST weights are downloaded automatically on first instantiation of the model.

If you would like to use CNN14: Download the pre-trained CNN14 weights `'Cnn14_mAP=0.431.pth'` from [here](https://zenodo.org/records/3987831) and provide the path as argument `--model.pretrained_ckpt <Path>` or add it to the config files.

### Datasets
We provide Lightning DataModules for ESC-50, NSynth and SpeechCommands. Please download the datasets from their respective sources and provide the path to the DataModules like so:
```bash
--data.dir <path-to-dataset>
```

## Basic Usage
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

## Example
To run adversarial training for PaSST on ESC-50 with 10 steps, 10 restarts, and epsilon = 0.5:

```bash
python -m run fit --config './training/base_config_esc50.yaml' \
 --data ESC50DataModule --data.dir <your-dataset-path> \
 --model PaSSTAdv --pgd_eps 0.5 --pgd_steps 10 --pgd_restarts 10
```
