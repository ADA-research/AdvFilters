from pytorch_lightning.cli import LightningCLI
from training.passt_vs_random import PasstAdv
from data.esc50_datamodule import ESC50DataModule

if __name__ == "__main__":
    cli = LightningCLI(PasstAdv, ESC50DataModule, save_config_callback=None)