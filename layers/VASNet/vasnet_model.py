# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
from layers.VASNet.layer_norm import  *
from replacement_functions import attn_msk


class SelfAttention(nn.Module):
    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)

        self.gradients = None  # placeholder for the gradients
        self.input_norm = None  # placeholder for the normalized input vectors

    def gradient_hook(self, grad) -> None:
        """ Setter method, for storing the computed gradients for the attention weights.

        :param torch.Tensor grad: A tensor with shape [T', T'] containing the gradients corresponding to the attn values
        """
        self.gradients = grad

    @staticmethod
    def get_entropy(logits) -> torch.Tensor:
        """ Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T]
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T]
        """
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        return _entropy / np.log(logits.shape[0])

    def forward(self, x, mask_attn=False, fragments=None):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1, 0))

        # Entropy is a measure of uncertainty: Higher value means less information.
        entropy = self.get_entropy(logits=logits)
        frames = entropy.size()
        ent_win = entropy.repeat(frames[0], 1)

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)

        if mask_attn:
            assert fragments is not None, f"Replace method 'attn_mask' needs at least a fragment to be provided!"
            att_weights_ = attn_msk(att_weights_, fragments)

        # register the hook and compute the input norm
        att_weights_.register_hook(self.gradient_hook)
        self.input_norm = torch.norm(V, p=2, dim=-1)

        div_h = torch.prod(1 - att_weights_, 1)
        div = div_h / torch.linalg.norm(div_h, ord=1)
        div_win = div.repeat(frames[0], 1)

        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1, 0), weights).transpose(1, 0)
        y = self.output_linear(y)

        return y, att_weights_.clone(), ent_win, div_win

    def get_attn_gradient(self) -> torch.Tensor:
        """ Getter method, to gain access for returning the attention gradients.

        :return: A tensor with shape [T', T'] containing the gradients corresponding to the attention values
        """
        return self.gradients

    def get_input_norm(self) -> torch.Tensor:
        """ InputNorm corresponds to the norm of transformed input vectors, ||v(x)||, in attention modules, where v(·)
            can be the value mappings in transformers.

        :return: A tensor with shape [T] containing the L2-normalized Value of attention mechanism
        """
        return self.input_norm


class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024   # CNN features size
        self.hidden_size = 1024

        self.attention = SelfAttention(input_size=self.m, output_size=self.m)
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)

    def forward(self, x, mask_attn=False, fragments=None):

        y, att_weights_, attn_entropy, attn_div = self.attention(x, mask_attn, fragments)

        y = y + x
        y = self.drop50(y)
        y = self.layer_norm_y(y)

        # Frame level importance score regression
        # Two layer NN
        y = self.ka(y)
        y = self.relu(y)
        y = self.drop50(y)
        y = self.layer_norm_ka(y)

        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)

        return y, att_weights_, attn_entropy, attn_div


if __name__ == "__main__":
    pass
