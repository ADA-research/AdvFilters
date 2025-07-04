from pytorch_lightning.cli import LightningCLI
from training.passt_masked_adv_train import PasstMaskedAdv
from training.openmic_datamodule import OpenMICDataModule

if __name__ == "__main__":
    LightningCLI(PasstMaskedAdv, OpenMICDataModule)