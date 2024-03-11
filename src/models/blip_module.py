import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L
from utils.blip_utils import *

class LitBlip(L.LightningModule):
    def __init__(self,                 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                net: torch.nn.Module
                ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        image, caption = batch
        loss = self.net(image, caption)
        
        return loss

    def trainning_epoch_end(self, outputs):
         pass

    def test_step(self, batch, batch_idx):
        result = []
        gt = []
        image, caption = batch      
        generated_caption = self.net.generate(image, sample=False)
        gt.extend(caption)
        result.extend(generated_caption)
        return result, gt
    
    def _common_step(self, batch, batch_idx):
        # training_step defines the train loop.
       
        return

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
