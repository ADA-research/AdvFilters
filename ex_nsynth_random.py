from pytorch_lightning.cli import LightningCLI
from training.passt_vs_random import PasstAdv
from data.nsynth_datamodule import NSynthDataModule

if __name__ == "__main__":
    cli = LightningCLI(PasstAdv, NSynthDataModule, save_config_callback=None)