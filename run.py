from pytorch_lightning.cli import LightningCLI
from training.passt_adv_train import PasstAdv
from training.cnn14_adv_train import CNN14Adv
from training.cnn14_vs_random import CNN14Random
from data.esc50_datamodule import ESC50DataModule
from data.nsynth_datamodule import NSynthDataModule
from data.speech_commands import SpeechCommandsDataModule

if __name__ == "__main__":
    cli = LightningCLI(save_config_callback=None)