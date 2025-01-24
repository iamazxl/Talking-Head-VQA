# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch

from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
import torch.nn as nn


def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]

    return frame_list


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)
            # print(x[0].shape)
            slow_feature = self.slow_avg_pool(x[0])
            fast_feature = self.fast_avg_pool(x[1])

            slow_feature = self.adp_avg_pool(slow_feature)
            fast_feature = self.adp_avg_pool(fast_feature)
            
        return slow_feature, fast_feature
