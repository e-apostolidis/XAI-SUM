# -*- coding: utf-8 -*-
import numpy as np
import math
from explanations import inference
from replacement_functions import slice_out, random_replace, input_msk
from utils import permitted_repl_methods, downsample
from data_loader import VideoData
from torch.nn import Module
import torch
from typing import Tuple, Dict
from scipy.stats import spearmanr, kendalltau, rankdata

permitted_sel_methods = ["single", "batch"]
permitted_corr_methods = ["spearman", "kendall"]
permitted_visual_masks = ["black-frame", "white-frame"]


def sanity_violation_test(model, loader, fragments_weight, repl_method, visual_token):
    """ Calculate the faithfulness violations, for the largest explanation weight of `fragments_weight` and the `model`
        prediction variance given a `repl_method`, for each video inside `loader`.

    :param Module model: The pretrained model to be explained
    :param VideoData loader: The data-loader of the current analysis experiment
    :param dict[str, np.ndarray] fragments_weight: A dictionary containing the fragment-level explanation weights for
                                 each video in loader
    :param str repl_method: The method used as replacement technique, see `permitted_repl_methods` for the options
    :param str visual_token: The visual mask for replacing the features of selected fragments, see `permitted_visual_masks` for the options
    :return: A dictionary containing the violations for each video
    """
    assert repl_method in permitted_repl_methods, f"Replacement method must be: {*permitted_repl_methods,}"

    videos_violations = {}
    for frame_features, fragments, video in loader:
        baseline, _, _, _, _, _ = inference(model=model, features=frame_features)  # The baseline scores
        fragments = downsample(fragments)
        fragments = fragments.numpy()

        curr_fragments_weight = fragments_weight[video]
        fragment = fragments[np.abs(curr_fragments_weight).argmax(), :]
        w_x = curr_fragments_weight[np.abs(curr_fragments_weight).argmax()]

        if repl_method == "slice-out":
            baseline = slice_out(baseline, [fragment])                 # Omit baseline prediction of the fragment(s)
            features_no_fragm = slice_out(frame_features, [fragment])  # Remove the features of the fragment(s)
            prediction, _, _, _, _, _ = inference(model, features_no_fragm)  # The newly predicted imp. scores
            fragments = fragments[:-1]
        elif repl_method == "attention-mask":
            prediction, _, _, _, _, _ = inference(model, frame_features, mask_attn=True, fragments=[fragment])
        elif repl_method == "input-mask":
            if visual_token == 'black-frame':
                visual_mask = torch.load('black.pt')
            elif visual_token == 'white-frame':
                visual_mask = torch.load('white.pt')
            features_masked_fragm = input_msk(frame_features, [fragment], token=visual_mask)
            prediction, _, _, _, _, _ = inference(model, features_masked_fragm)
        else:   # repl_method == "random"
            features_random_fragm = random_replace(frame_features, [fragment], size=512)
            prediction, _, _, _, _, _ = inference(model, features_random_fragm)

        # Compute what is a violation for a given video
        fragment_diff = np.zeros(fragments.shape[0])
        for count, fragment in enumerate(fragments):
            fragment_baseline = baseline[fragment[0]:fragment[1]].mean()
            fragment_prediction = prediction[fragment[0]:fragment[1]].mean()
            fragment_diff[count] = fragment_baseline - fragment_prediction  # Delta_E (fragment-level)

        violations = ((fragment_diff * w_x) < 0).sum() / fragments.shape[0] * 100
        videos_violations[video] = violations > 50.0

    return videos_violations


def comprehensiveness(pred_score, base_score) -> float:
    """ Compare the output vector (i.e. importance scores) for the baseline (entire) video and the version which omits a
        shot of the video, using one of the replacement functions.

    :param np.ndarray pred_score: An array with shape [T'], containing the 'new' predicted scores
    :param np.ndarray base_score: An array with shape [T'], containing the -baseline- imp. scores for the entire video
    :return: float: The computed comprehensiveness score
    """
    c, p = kendalltau(rankdata(base_score), rankdata(pred_score))
    return 1.0 - c


def rank_correlation(model, loader, fragments_weight, repl_method, visual_token, corr_method="spearman") -> Dict[str, float]:
    """ Calculate the `corr_method` rank correlation, between the magnitude of the `fragments_weight` and the `model`
        prediction variance given a `repl_method`, for each video inside `loader`.

    :param Module model: The pretrained model to be explained
    :param VideoData loader: The data-loader of the current analysis experiment
    :param dict[str, np.ndarray] fragments_weight: A dictionary containing the fragment-level explanation weights for
                                 each video in loader
    :param str repl_method: The method used as replacement technique, see `permitted_repl_methods` for the options
    :param str visual_token: The visual mask for replacing the features of selected fragments, see `permitted_visual_masks` for the options
    :param str corr_method: The method used for calculate the correlation, see `permitted_corr_methods` for the options
    :return: A dictionary containing the rank correlation for each video
    """
    assert corr_method in permitted_corr_methods, f"Correlation method must be: {*permitted_corr_methods,}"
    # print("------- Only applicable for `single` selection method of video fragments! -------")

    videos_ranking_correlation = {}
    for frame_features, fragments, video in loader:
        baseline, _, _, _, _, _ = inference(model=model, features=frame_features)  # The baseline scores
        fragments = downsample(fragments)
        fragments = fragments.numpy()
        curr_fragments_weight = fragments_weight[video]

        fragments_comprehension = []
        for fragment in fragments:
            if repl_method == "slice-out":                                     # Omit the shot entirely
                base_score_no_fragm = slice_out(baseline, [fragment])          # Omit baseline prediction of the fragment(s)
                features_no_fragm = slice_out(frame_features, [fragment])      # Remove the features of the fragment(s)
                score_no_fragm, _, _, _, _, _ = inference(model, features_no_fragm)  # The newly predicted imp. scores

                fragments_comprehension.append(comprehensiveness(score_no_fragm, base_score_no_fragm.numpy()))
            elif repl_method == "attention-mask":
                base_score = baseline.copy()
                score_masked_fragm, _, _, _, _, _ = inference(model, frame_features, mask_attn=True, fragments=[fragment])

                comp = comprehensiveness(score_masked_fragm, base_score)
                fragments_comprehension.append(comp)
            elif repl_method == "input-mask":
                base_score = baseline.copy()
                if visual_token == 'black-frame':
                    visual_mask = torch.load('black.pt')
                elif visual_token == 'white-frame':
                    visual_mask = torch.load('white.pt')
                features_masked_fragm = input_msk(frame_features, [fragment], token=visual_mask)
                score_masked_fragm, _, _, _, _, _ = inference(model, features_masked_fragm)

                fragments_comprehension.append(comprehensiveness(score_masked_fragm, base_score))
            elif repl_method == "random":
                base_score = baseline.copy()
                features_random_fragm = random_replace(frame_features, [fragment], size=512)
                score_random_fragm, _, _, _, _, _ = inference(model, features_random_fragm)

                fragments_comprehension.append(comprehensiveness(score_random_fragm, base_score))

        comprehension = np.abs(np.array(fragments_comprehension))
        curr_fragments_weight = np.abs(curr_fragments_weight)
        if corr_method == "spearman":
            curr_corr_coeff, _ = spearmanr(comprehension, curr_fragments_weight)
        else:
            curr_corr_coeff, _ = kendalltau(rankdata(comprehension), rankdata(curr_fragments_weight))
        videos_ranking_correlation[video] = curr_corr_coeff

    return videos_ranking_correlation


def get_measures(model, loader, ranked_fragments, repl_method, repl_fragments, visual_token) -> Tuple[float, float, float]:
    """
    :param torch.nn.Module model: The pretrained model to be explained
    :param VideoData loader: The data-loader of the current analysis experiment
    :param dict[str, np.ndarray] ranked_fragments: Dictionary of arrays with the indices of the ranked video fragments
                                 (in ascending order) using their attention-based significance
    :param str repl_method: The method used as replacement technique, see `permitted_repl_methods` for the options
    :param str repl_fragments: The amount of replaced fragments, see `permitted_sel_methods` for the options
    :param str visual_token: The visual mask for replacing the features of selected fragments, see `permitted_visual_masks` for the options
    :return: A tuple containing:
                A float consisting of the average computed `disc_minus` scores for the test videos
                A float consisting of the average computed `disc_plus` scores for the test videos
    """
    assert repl_method in permitted_repl_methods, f"Replacement method must be: {*permitted_repl_methods,}"
    assert repl_fragments in permitted_sel_methods, f"Selection method must be: {*permitted_sel_methods,}"

    iterations = None
    if repl_fragments == "single":
        iterations = range(5)
    elif repl_fragments == "batch":
        iterations = [0.01, 0.05, 0.1, 0.15, 0.2]

    disc_plus, disc_minus = [], []
    sanity = []
    for frame_features, fragments, video in loader:
        baseline, _, _, _, _, _ = inference(model=model, features=frame_features)  # The baseline scores
        fragments = downsample(fragments)
        fragments = fragments.numpy()
        curr_fragment_ranking = ranked_fragments[video]

        video_disc_plus, video_disc_minus = [], []
        for _iter in iterations:
            comp_top, comp_less = None, None
            top_scored_fragments, less_scored_fragments = [], []

            # Get the top (less) scored fragments
            if repl_fragments == "single":
                curr_top_fragm_ids = [curr_fragment_ranking[_iter]]
                curr_less_fragm_ids = [curr_fragment_ranking[::-1][_iter]]        # Read the ranking in reverse order
            else:
                num_of_picks = math.ceil(_iter * curr_fragment_ranking.shape[0])

                curr_top_fragm_ids = curr_fragment_ranking[:num_of_picks]
                curr_top_fragm_ids = -np.sort(-curr_top_fragm_ids)                # Ordering to help with slice-out

                curr_less_fragm_ids = curr_fragment_ranking[::-1][:num_of_picks]  # Read the ranking in reverse order
                curr_less_fragm_ids = -np.sort(-curr_less_fragm_ids)

            top_scored_fragments.extend(fragments[curr_top_fragm_ids, :])
            less_scored_fragments.extend(fragments[curr_less_fragm_ids, :])

            if repl_method == "slice-out":  # Omit the shot entirely
                base_score_no_top = slice_out(baseline, top_scored_fragments)        # Omit baseline prediction of the fragment(s)
                features_no_top = slice_out(frame_features, top_scored_fragments)    # Remove the features of the fragment(s)
                score_no_top, _, _, _, _, _ = inference(model, features_no_top)            # The newly predicted imp. scores

                base_score_no_less = slice_out(baseline, less_scored_fragments)      # Omit baseline prediction of the fragment(s)
                features_no_less = slice_out(frame_features, less_scored_fragments)  # Remove the features of the fragment(s)
                score_no_less, _, _, _, _, _ = inference(model, features_no_less)          # The newly predicted imp. scores

                comp_less = comprehensiveness(score_no_less, base_score_no_less.numpy())
                comp_top = comprehensiveness(score_no_top, base_score_no_top.numpy())
            elif repl_method == "attention-mask":
                base_score = baseline.copy()

                score_masked_top, _, _, _, _, _ = inference(model, frame_features,
                                                      mask_attn=True, fragments=top_scored_fragments)
                score_masked_less, _, _, _, _, _ = inference(model, frame_features,
                                                       mask_attn=True, fragments=less_scored_fragments)

                comp_less = comprehensiveness(score_masked_less, base_score)
                comp_top = comprehensiveness(score_masked_top, base_score)
                if np.isnan(comp_less):
                    print("stop")
            elif repl_method == "input-mask":
                base_score = baseline.copy()
                if visual_token == 'black-frame':
                    visual_mask = torch.load('black.pt')
                elif visual_token == 'white-frame':
                    visual_mask = torch.load('white.pt')
                features_masked_top = input_msk(frame_features, top_scored_fragments, token=visual_mask)
                score_masked_top, _, _, _, _, _ = inference(model, features_masked_top)

                features_masked_less = input_msk(frame_features, less_scored_fragments, token=visual_mask)
                score_masked_less, _, _, _, _, _ = inference(model, features_masked_less)

                comp_less = comprehensiveness(score_masked_less, base_score)
                comp_top = comprehensiveness(score_masked_top, base_score)
            elif repl_method == "random":
                base_score = baseline.copy()

                features_random_top = random_replace(frame_features, top_scored_fragments, size=512)
                score_random_top, _, _, _, _, _ = inference(model, features_random_top)

                features_random_less = random_replace(frame_features, less_scored_fragments, size=512)
                score_random_less, _, _, _, _, _ = inference(model, features_random_less)

                comp_less = comprehensiveness(score_random_less, base_score)
                comp_top = comprehensiveness(score_random_top, base_score)

            video_disc_minus.append(comp_less)
            video_disc_plus.append(comp_top)

        disc_minus.append(np.mean(video_disc_minus))
        disc_plus.append(np.mean(video_disc_plus))

        if disc_minus[-1] < disc_plus[-1]:
            sanity.append(1.0)
        else:
            sanity.append(0.0)

    disc_minus = np.round(np.array(disc_minus).mean(), 8)
    disc_plus = np.round(np.array(disc_plus).mean(), 8)

    sanity = np.round((1.0 - np.array(sanity).mean()), 3)
    return disc_minus, disc_plus, sanity
