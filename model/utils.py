# -*- coding: utf-8 -*-
import torch
import numpy as np

permitted_repl_methods = ["slice-out", "attention-mask", "input-mask", "random"]


def downsample(shots) -> torch.Tensor:
    """ Transcode the frame IDs of the shots from 30 to 2 fps.

    :param torch.Tensor shots: A tensor with shape [T, 2] containing the frame IDs for the 30 fps video
    :return: A tensor with shape [T, 2] containing the frame IDs for the (down-sampled) 2 fps video
    """
    downsampled_shots = torch.floor(shots / 15).to(dtype=torch.long)
    return downsampled_shots


def normalize(arr) -> np.ndarray:
    """ Normalize the given array, by subtracting the min and dividing with the max value.

    :param np.ndarray arr: The array to be normalized
    :return: The normalized array, with min = 0 and max = 1
    """
    arr -= arr.min(initial=np.inf)
    arr /= arr.max(initial=-np.inf)
    return arr


def masked_softmax(attn_weights) -> torch.Tensor:
    """ Compute the softmax of the non-zero values of the attention matrix. Differentiality must be retained!!

    :param torch.Tensor attn_weights: The processed attention energies, that must be transformed back to weights again
    :return: A tensor with shape [T, T] softmax-ed by its last dim
    """
    attn_weights = attn_weights.masked_fill(attn_weights == 0, -float('inf'))
    return torch.nn.Softmax(dim=-1)(attn_weights)
