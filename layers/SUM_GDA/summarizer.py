# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from layers.SUM_GDA.attention import SelfAttention
from typing import Tuple


class SUM_GDA(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        """ Class wrapping the SUM-GDA model; its key modules and parameters.
        
        :param int input_size: The expected input feature size.
        :param int output_size: The produced output feature size.
        """
        super(SUM_GDA, self).__init__()

        self.attention = SelfAttention(input_size=input_size, output_size=output_size)
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(in_features=self.linear_1.out_features, out_features=1)

        self.drop = nn.Dropout(p=0.6)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(normalized_shape=self.linear_1.out_features, eps=1e-6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features, mask_attn=False, fragments=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Produce frames importance scores from the frame features, using the SUM-GDA model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by 
        using the pool5 layer of GoogleNet.
        :param bool mask_attn: A boolean flag controlling the masking (zeroing) of the attention weights
        :param None | list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending
                                       ordered [start, end] indices of the fragments to be masked, if requested
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """

        weighted_value, attn_weights, attn_entropy, attn_div = self.attention(frame_features, mask_attn, fragments)
        weighted_value = self.drop(weighted_value)
        weighted_value = self.norm_y(weighted_value)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(weighted_value)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.view(1, -1)

        return y, attn_weights, attn_entropy, attn_div


if __name__ == '__main__':
    pass
