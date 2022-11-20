from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter
import random

from utils import concat_all_gather


class CSFDA_Model(nn.Module):
    """
    Student and Teacher Models
    """

    def __init__(
        self,
        src_model,
        ema_model,
        m=0.98,                 ## Use 0.95 for DomainNet, 0.98 or more for VISDA-C 
        checkpoint_path=None,
    ):
        """
        dim: feature dimension (default: 128)
        m: EMA coefficient teacher model (default: 0.999)
        """
        super(CSFDA_Model, self).__init__()

        self.m = m
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.ema_model = ema_model

        # create the fc heads
        feature_dim = src_model.output_dim

        # freeze key model
        self.ema_model.requires_grad_(False)
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            
            ## Get rid of 'module.' prefix brought by DDP
            name = name[len("module.") :] if name.startswith("module.") else name
            state_dict[name] = param
        
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    @torch.no_grad()
    def _ema_model_update(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.src_model.parameters(), self.ema_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    
    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Input:
            im_q: 1st batch of augmented images
            im_k: 2nd batch of aumented images
        Output:
            Features and logits
        """

        ## Compute query features
        feats_q, logits_q = self.src_model(im_q, return_feats=True)

        if cls_only:
            return feats_q, logits_q

        ## EMA update of the Teacher Model
        with torch.no_grad():                    
            self._ema_model_update()  

        # dequeue and enqueue will happen outside
        return feats_q, logits_q



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
