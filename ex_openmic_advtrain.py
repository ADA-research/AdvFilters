from pytorch_lightning.cli import LightningCLI
from training.passt_masked_adv_train import PasstMaskedAdv
from data.openmic_datamodule import OpenMICDataModule

if __name__ == "__main__":
    cli = LightningCLI(PasstMaskedAdv, OpenMICDataModule, save_config_callback=None)