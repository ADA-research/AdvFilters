from pytorch_lightning.cli import LightningCLI
from training.passt_vs_random import PasstAdv
from data.speech_commands import SpeechCommandsDataModule

if __name__ == "__main__":
    cli = LightningCLI(PasstAdv, SpeechCommandsDataModule, save_config_callback=None)