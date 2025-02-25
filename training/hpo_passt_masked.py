import numpy as np
from pytorch_lightning.cli import LightningCLI 
from sklearn.metrics import average_precision_score
import torch

from openmic_datamodule import OpenMICDataModule
from hpo_passt import PasstHPO
from utils import masked_mean_average_precision

def masked_bce_loss(y_hat, y_true, mask):
    samples_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_hat, y_true, reduction="none"
    )
    samples_loss = mask.float() * samples_loss
    loss = samples_loss.mean()
    samples_loss = samples_loss.detach()
    return loss

class PasstHPOMasked(PasstHPO):
    def __init__(self, 
                 pretrained_arch:str = None,
                 num_classes = 20, 
                 lr = 0.00001, 
                 s_patchout_t = 40, 
                 s_patchout_f = 4, 
                 *args, **kwargs):
        super().__init__(pretrained_arch, num_classes, lr, s_patchout_t, s_patchout_f, *args, **kwargs)
        self.loss = masked_bce_loss
        
    def forward(self, x):
        return super().forward(x)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y, mask)
        self.log("test_loss", loss)
        return {"y_hat": y_hat, "y": y, "mask": mask}

    def test_epoch_end(self, outputs):
        y_hats = []
        y_trues = []
        masks = []
        for res in outputs:
            y_hats.append(torch.sigmoid(res["y_hat"]).cpu().numpy())
            y_trues.append(res["y"].cpu().numpy())
            masks.append(res["mask"].cpu().numpy())
        y_hats = np.vstack(y_hats)
        y_trues = np.vstack(y_trues)
        masks = np.vstack(masks)
        mAP = masked_mean_average_precision(y_trues, y_hats, masks)
        self.log("mAP", mAP)

if __name__ == "__main__":
    cli = LightningCLI(PasstHPOMasked, OpenMICDataModule)