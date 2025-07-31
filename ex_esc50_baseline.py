from pytorch_lightning.cli import LightningCLI
from training.passt import Passt
from data.esc50_datamodule import ESC50DataModule

if __name__ == "__main__":
    cli = LightningCLI(Passt, ESC50DataModule, save_config_callback=None)