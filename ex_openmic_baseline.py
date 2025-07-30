from pytorch_lightning.cli import LightningCLI
from training.passt_masked import PasstMasked
from data.openmic_datamodule import OpenMICDataModule

if __name__ == "__main__":
    cli = LightningCLI(PasstMasked, OpenMICDataModule, save_config_callback=None)