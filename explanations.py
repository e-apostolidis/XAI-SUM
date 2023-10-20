# -*- coding: utf-8 -*-
import numpy as np
import torch
from typing import Tuple, Optional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
M = 60  # Block size used in CA-SUM


def raw_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The raw attention values are used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the raw attention values, respectively
    """
    imp_score, attn_weights, _, _, _, _ = inference(model, features)

    return imp_score, attn_weights


def ent_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The entropy of the raw attention values is used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the entropy of the attention values, respectively
    """
    imp_score, _, _, _, attn_entropy, _ = inference(model, features)

    return imp_score, attn_entropy


def div_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The diversity of the raw attention values is used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the diversity of the attention values, respectively
    """
    imp_score, _, _, _, _, attn_div = inference(model, features)

    return imp_score, attn_div


def grad_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The attention values x grad(attention values) are used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the attention values multiplied with the
             corresponding gradients, respectively
    """
    imp_score, attn_weights, attn_grads, _, _, _ = inference(model, features, uses_gradient=True)
    weights = attn_weights * attn_grads

    return imp_score, weights


def grad_of_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The attention values x grad(attention values) are used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the attention values multiplied with the
             corresponding gradients, respectively
    """
    imp_score, attn_weights, attn_grads, _, _, _ = inference(model, features, uses_gradient=True)
    weights = attn_grads

    return imp_score, weights


def input_norm_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The attention values x norm(attention.Value) are used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the attention values multiplied with the
             L2-normalized Value of the attention mechanism, respectively
    """
    imp_score, attn_weights, _, input_norm, _, _ = inference(model, features, uses_input_norm=True)
    input_norm = input_norm.reshape(1, -1)
    input_norm = np.tile(input_norm.transpose(), (1, imp_score.shape[0]))  # repeat the value row-wise
    weights = attn_weights * input_norm

    return imp_score, weights


def input_norm_grad_attn(model, features) -> Tuple[np.ndarray, np.ndarray]:
    """ The attention values x norm(attention.Value) are used as an explanation (weight).

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :return: A tuple of arrays containing the predicted importance scores and the attention values multiplied with the
             L2-normalized Value of the attention mechanism, respectively
    """
    imp_score, attn_weights, attn_grads, input_norm, _, _ = inference(model, features, uses_gradient=True, uses_input_norm=True)
    input_norm = input_norm.reshape(1, -1)
    input_norm = np.tile(input_norm.transpose(), (1, imp_score.shape[0]))  # repeat the value row-wise
    weights = attn_weights * input_norm * attn_grads

    return imp_score, weights


def inference(model, features, uses_gradient=False, uses_input_norm=False, mask_attn=False,
              fragments=None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],]:
    """ Run the pretrained model over a video, to get the importance scores produced for its frames.

    :param torch.nn.Module model: The pretrained model to be explained
    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :param bool uses_gradient: A boolean variable controlling if the grad for the attn_weights will be computed
    :param bool uses_input_norm: A boolean variable controlling if the L2-normalized Value will be computed
    :param bool mask_attn: A boolean flag controlling the masking (zeroing) of the attention weights
    :param None | list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending
                                   ordered [start, end] indices of the fragments to be masked, if requested
    :return: A tuple containing:
              An array with shape [T], containing the frame-level importance scores produced by the model
              A 2d array with shape [T, T] holding the weights of the self-attention mechanism
              if requested: A 2d array with shape [T, T] consisting of the grads (output w.r.t the weights)
              if requested: An array with shape [T] containing the L2-normalized Value of the attention mechanism
    """
    attn_grads = None
    input_norm = None
    model.eval()

    required_size = features.shape[0]
    # Make sure that feature is at least M frame long (same as the block selected)
    if features.shape[0] <= M:
        remaining = M - features.shape[0] + 1
        rem_tensor = torch.zeros(remaining, features.shape[1])
        features = torch.cat((features, rem_tensor))

    features = features.to(device)
    imp_score, attn_weights, attn_entropy, attn_div = model(features, mask_attn, fragments)

    # The padding is affecting both imp_scores and attention_weights (grad and input norm)
    imp_score = imp_score.squeeze(0)[:required_size]
    attn_weights = attn_weights[:required_size, :required_size]

    if uses_gradient:
        # from here: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
        model.zero_grad()
        model.attention.zero_grad()

        imp_score.sum().backward(retain_graph=True)
        attn_grads = model.attention.get_attn_gradient().cpu().numpy()
        attn_grads = attn_grads[:required_size, :required_size]

    if uses_input_norm:
        input_norm = model.attention.get_input_norm().detach().cpu().numpy()
        input_norm = input_norm[:required_size]

    imp_score = imp_score.detach().cpu().numpy()
    attn_weights = attn_weights.detach().cpu().numpy()
    attn_entropy = attn_entropy.detach().cpu().numpy()
    attn_div = attn_div.detach().cpu().numpy()

    return imp_score, attn_weights, attn_grads, input_norm, attn_entropy, attn_div
