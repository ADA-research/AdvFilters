import numpy as np
from pytorch_lightning.cli import LightningCLI 
import torch

from data.openmic import OpenMICDataModule
from training.passt import Passt
from data.utils import masked_mean_average_precision

def masked_bce_loss(y_hat, y_true, mask):
    samples_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_hat, y_true, reduction="none"
    )
    samples_loss = mask.float() * samples_loss
    loss = samples_loss.mean()
    samples_loss = samples_loss.detach()
    return loss

class PasstMasked(Passt):
    def __init__(self, 
                 pretrained_arch:str = None,
                 num_classes = 20, 
                 lr = 0.00001, 
                 s_patchout_t = 40, 
                 s_patchout_f = 4, 
                 *args, **kwargs):
        super().__init__(pretrained_arch, num_classes, lr, s_patchout_t, s_patchout_f, *args, **kwargs)
        self.loss = masked_bce_loss
        
        # For aggregating test metrics
        self.y_hats = []
        self.y_trues = []
        self.masks = []
        
    def forward(self, x):
        return super().forward(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("test/loss", loss)
        self.y_hats.append(y_hat.cpu().numpy())
        self.y_trues.append(y.cpu().numpy())
        self.masks.append(mask.cpu().numpy())
        return {"y_hat": y_hat, "y": y, "mask": mask}

    def on_test_epoch_end(self):
        y_hats = np.vstack(self.y_hats)
        y_trues = np.vstack(self.y_trues)
        masks = np.vstack(self.masks)
        mAP = masked_mean_average_precision(y_trues, y_hats, masks)
        self.log("test/mAP", mAP)
        self.y_hats = []
        self.y_trues = []
        self.masks = []
        
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("val/loss", loss)
        self.y_hats.append(y_hat.cpu().numpy())
        self.y_trues.append(y.cpu().numpy())
        self.masks.append(mask.cpu().numpy())
        return {"y_hat": y_hat, "y": y, "mask": mask}

    def on_validation_epoch_end(self):
        y_hats = np.vstack(self.y_hats)
        y_trues = np.vstack(self.y_trues)
        masks = np.vstack(self.masks)
        mAP = masked_mean_average_precision(y_trues, y_hats, masks)
        self.log("val/mAP", mAP)
        self.y_hats = []
        self.y_trues = []
        self.masks = []

if __name__ == "__main__":
    cli = LightningCLI(PasstMasked, OpenMICDataModule)