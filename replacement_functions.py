# -*- coding: utf-8 -*-
import torch
import numpy as np
import math
from explanations import M
from utils import masked_softmax


def slice_out(tensor, fragments) -> torch.Tensor:
    """ Remove the `fragments` from the `tensor`, to simulate a video w/o this shot.

    :param torch.Tensor | np.ndarray tensor: A(n) tensor (array) with shape [T, *] containing the initial data
    :param list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending ordered
                                       [start, end] indices of the fragments to be removed
    :return: A tensor with shape [T-sum(len(fragments[:])), *] containing the processed data, w/o the selected fragments
    """
    if type(tensor) is np.ndarray:
        tensor = torch.from_numpy(tensor)

    processed_tensor = tensor.clone()
    for fragment in fragments:  # Remove the furthest fragment(s), so the previous ones are not affected by the removal
        processed_tensor = torch.cat((processed_tensor[:fragment[0]], processed_tensor[fragment[1]:]), dim=0)

    return processed_tensor


def attn_msk(attn_weights, fragments) -> torch.Tensor:
    """ Zero the attention values in `attn_weights` that corresponds to the frames inside `fragments`.

    :param torch.Tensor attn_weights: A tensor of shape [T, T] containing the (blocked) attention matrix
    :param list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending ordered
                                       [start, end] indices of the fragments that will have its attn values zeroed
    :return: A tensor with shape [T, T], with zeroed attention values inside `fragments`
    """
    __altered = False
    __block_num = math.floor(attn_weights.shape[0] / M)
    last_block_len = attn_weights.shape[0] - (__block_num * M)

    processed_weights = attn_weights.clone()
    for fragment in fragments:
        fragment_size = fragment[1] - fragment[0]
        if fragment[1] >= attn_weights.shape[0] and fragment_size > last_block_len:
            continue
        processed_weights[fragment[0]:fragment[1],
                          fragment[0]:fragment[1]] = 0.
        __altered = True

    if __altered:
        processed_weights = masked_softmax(processed_weights)

    return processed_weights


def input_msk(features, fragments, token=0.) -> torch.Tensor:
    """ Mask the feature vectors (aka equal to token) corresponding to the frames inside the `fragments`.

    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :param list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending ordered
                                       [start, end] indices of the fragments to be masked
    :param float | torch.Tensor token: An int or a tensor with shape [1, input_size] holding the mask value to be used
    :return: A tensor with shape [T, input_size], with masked feature vectors inside `fragments`
    """
    processed_features = features.clone()
    for fragment in fragments:
        processed_features[fragment[0]:fragment[1], :] = token

    return processed_features


def random_replace(features, fragments, size=512) -> torch.Tensor:
    """ Change `size` random bits of the feature vector of each frame inside `fragments` with randomly picked bits (on
        the same "column") of randomly picked frames outside 'fragments'.

    :param torch.Tensor features: A tensor with shape [T, input_size], containing the video's frame features
    :param list[np.ndarray] fragments: A list (of len N) of arrays with shape [2] containing the descending ordered
                                       [start, end] indices of the fragments to be randomly mutated
    :param int size: The size of the mutation vector
    :return: A tensor with shape [T, input_size] with randomly mutated feature vectors inside `fragments`
    """

    processed_features = features.clone()
    sampling_features = features.clone()
    for fragment in fragments:
        sampling_features = torch.cat((sampling_features[:fragment[0], :],
                                       sampling_features[fragment[1]:, :]), dim=0)

    for fragment in fragments:
        length = features.shape[1]
        indices = np.random.choice(a=length, size=size, replace=False)
        indices.sort()
        start_frame, end_frame = fragment[0], min(features.shape[0], fragment[1])
        for frame_id in range(start_frame, end_frame):
            for index in indices:
                chosen_frame = torch.randint(low=0, high=sampling_features.shape[0], size=(1,))

                processed_features[frame_id, index] = sampling_features[chosen_frame, index]

    return processed_features
