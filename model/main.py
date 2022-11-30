# -*- coding: utf-8 -*-
import torch
import numpy as np
from layers.summarizer import CA_SUM
from os import listdir
from os.path import isfile, join
from configs import get_config
from data_loader import get_loader
from explanations import raw_attn, grad_attn, grad_of_attn, input_norm_attn, input_norm_grad_attn
from explanation_metrics import get_measures, rank_correlation, sanity_violation_test
from utils import downsample
from prettytable import PrettyTable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExplanationEvalMeter:
    def __init__(self, name, loader, model):
        self.name = name
        self.loader = loader
        self.model = model

        self.curr_frames_weight = None
        self.curr_fragments_weight = None
        self.fragments_weight = {}
        self.ranked_fragments = {}
        self.videos_violation = {}

    def get_frames_weight(self, _weights):
        self.curr_frames_weight = _weights.diagonal()

    def get_fragments_weight(self, _fragments):
        self.curr_fragments_weight = np.zeros(_fragments.shape[0])  # fragment-level explanation weights
        for count, fragment in enumerate(_fragments):
            self.curr_fragments_weight[count] = self.curr_frames_weight[fragment[0]:fragment[1]].mean()

    def get_ranking(self, _video):
        fragments_ranking = self.curr_fragments_weight.argsort()[::-1]  # rank fragments in descending order
        self.ranked_fragments[_video] = fragments_ranking
        self.fragments_weight[_video] = self.curr_fragments_weight

    def get_violations(self, _repl_method, _visual_token):
        self.videos_violation = sanity_violation_test(self.model, self.loader, self.fragments_weight, _repl_method, _visual_token)


if __name__ == "__main__":

    # Get the experimental settings
    config = get_config()
    dataset = config.dataset                    # SumMe, TVSum
    repl_method = config.replacement_method     # slice-out, input-mask, random, attention-mask
    repl_fragments = config.replaced_fragments  # batch, single
    visual_mask = config.visual_mask

    # Define the lists keeping the results for each data split
    att_disc_minus_all, norm_att_disc_minus_all, grad_att_disc_minus_all, grad_of_att_disc_minus_all, norm_grad_att_disc_minus_all = [], [], [], [], []
    att_disc_plus_all, norm_att_disc_plus_all, grad_att_disc_plus_all, grad_of_att_disc_plus_all, norm_grad_att_disc_plus_all = [], [], [], [], []
    att_sanity_test_all, norm_att_sanity_test_all, grad_att_sanity_test_all, grad_of_att_sanity_test_all, norm_grad_att_sanity_test_all = [], [], [], [], []
    att_rank_corr_all, norm_att_rank_corr_all, grad_att_rank_corr_all, grad_of_att_rank_corr_all, norm_grad_att_rank_corr_all = [], [], [], [], []

    for split_id in range(5):
        # Model data
        model_path = f".../XAI-SUM/data/pretrained_models/{dataset}/split{split_id}"
        model_file = [f for f in listdir(model_path) if isfile(join(model_path, f))]

        # Create model according to the configuration reported in the relevant paper
        trained_model = CA_SUM(input_size=1024, output_size=1024, block_size=60).to(device)
        trained_model.load_state_dict(torch.load(join(model_path, model_file[-1])))

        # Get a data loader containing the video attributes in need
        video_loader = get_loader(video_type=dataset, split_index=split_id, step=20)

        # Create objects for ranking video fragments according to the different explanation signals
        raw_attn_eval = ExplanationEvalMeter(name="raw-attn", loader=video_loader, model=trained_model)
        grad_attn_eval = ExplanationEvalMeter(name="grad-attn", loader=video_loader, model=trained_model)
        grad_of_attn_eval = ExplanationEvalMeter(name="grad-of-attn", loader=video_loader, model=trained_model)
        norm_attn_eval = ExplanationEvalMeter(name="norm-attn", loader=video_loader, model=trained_model)
        norm_grad_attn_eval = ExplanationEvalMeter(name="norm-grad-attn", loader=video_loader, model=trained_model)

        for frame_features, fragments, video in video_loader:
            fragments = downsample(fragments)

            # Explanation mask: Raw Attention
            _, weights = raw_attn(trained_model, frame_features)
            raw_attn_eval.get_frames_weight(weights)
            raw_attn_eval.get_fragments_weight(fragments)
            raw_attn_eval.get_ranking(video)

            # Explanation mask: Attention * Gradient
            _, weights = grad_attn(trained_model, frame_features)
            grad_attn_eval.get_frames_weight(weights)
            grad_attn_eval.get_fragments_weight(fragments)
            grad_attn_eval.get_ranking(video)

            # Explanation mask: Gradient of attention
            _, weights = grad_of_attn(trained_model, frame_features)
            grad_of_attn_eval.get_frames_weight(weights)
            grad_of_attn_eval.get_fragments_weight(fragments)
            grad_of_attn_eval.get_ranking(video)

            # Explanation mask: Attention * Input_Norm
            _, weights = input_norm_attn(trained_model, frame_features)
            norm_attn_eval.get_frames_weight(weights)
            norm_attn_eval.get_fragments_weight(fragments)
            norm_attn_eval.get_ranking(video)

            # Explanation mask: Attention * Input_Norm * Gradient
            _, weights = input_norm_grad_attn(trained_model, frame_features)
            norm_grad_attn_eval.get_frames_weight(weights)
            norm_grad_attn_eval.get_fragments_weight(fragments)
            norm_grad_attn_eval.get_ranking(video)

        # Compute Discoverability+, Discoverability- and Sanity Violation
        # Explanation mask: Raw Attention
        att_disc_minus, att_disc_plus, att_sanity_test = get_measures(model=trained_model, loader=video_loader,
                                                                                  ranked_fragments=raw_attn_eval.ranked_fragments,
                                                                                  repl_method=repl_method,
                                                                                  repl_fragments=repl_fragments,
                                                                                  visual_token=visual_mask)

        # Explanation mask: Attention * Gradient
        grad_att_disc_minus, grad_att_disc_plus, grad_att_sanity_test = get_measures(model=trained_model,
                                                                                                 loader=video_loader,
                                                                                                 ranked_fragments=grad_attn_eval.ranked_fragments,
                                                                                                 repl_method=repl_method,
                                                                                                 repl_fragments=repl_fragments,
                                                                                                 visual_token=visual_mask)
        # Explanation mask: Gradient of Attention
        grad_of_att_disc_minus, grad_of_att_disc_plus, grad_of_att_sanity_test = get_measures(
            model=trained_model, loader=video_loader,
            ranked_fragments=grad_of_attn_eval.ranked_fragments,
            repl_method=repl_method,
            repl_fragments=repl_fragments,
            visual_token=visual_mask)

        # Explanation mask: Attention * Input_Norm
        norm_att_disc_minus, norm_att_disc_plus, norm_att_sanity_test = get_measures(model=trained_model,
                                                                                                 loader=video_loader,
                                                                                                 ranked_fragments=norm_attn_eval.ranked_fragments,
                                                                                                 repl_method=repl_method,
                                                                                                 repl_fragments=repl_fragments,
                                                                                                 visual_token=visual_mask)

        # Explanation mask: Attention * Input_Norm * Gradient
        norm_grad_att_disc_minus, norm_grad_att_disc_plus, norm_grad_att_sanity_test = get_measures(
            model=trained_model, loader=video_loader,
            ranked_fragments=norm_grad_attn_eval.ranked_fragments,
            repl_method=repl_method,
            repl_fragments=repl_fragments,
            visual_token=visual_mask)

        # Compute Rank Correlation (only for one-by-one replacements)
        if repl_fragments == "single":
            # Explanation mask: Raw Attention
            att_rank_correlation_dict = rank_correlation(model=trained_model, loader=video_loader,
                                                         fragments_weight=raw_attn_eval.fragments_weight,
                                                         repl_method=repl_method, visual_token=visual_mask,
                                                         corr_method="spearman")

            # Fisher-z transformation for computing the average rank correlation
            att_rank_correlation_values = np.array(list(att_rank_correlation_dict.values()))
            z_values = np.arctanh(att_rank_correlation_values)
            mean_z = np.mean(z_values)
            att_rank_corr = np.tanh(mean_z)
            att_rank_corr = np.round(att_rank_corr, 3)

            # Explanation mask: Attention * Gradient
            grad_att_rank_correlation_dict = rank_correlation(model=trained_model, loader=video_loader,
                                                              fragments_weight=grad_attn_eval.fragments_weight,
                                                              repl_method=repl_method, visual_token=visual_mask,
                                                              corr_method="spearman")

            # Fisher-z transformation for computing the average rank correlation
            grad_att_rank_correlation_values = np.array(list(grad_att_rank_correlation_dict.values()))
            z_values = np.arctanh(grad_att_rank_correlation_values)
            mean_z = np.mean(z_values)
            grad_att_rank_corr = np.tanh(mean_z)
            grad_att_rank_corr = np.round(grad_att_rank_corr, 3)

            # Explanation mask: Gradient of Attention
            grad_of_att_rank_correlation_dict = rank_correlation(model=trained_model, loader=video_loader,
                                                                 fragments_weight=grad_of_attn_eval.fragments_weight,
                                                                 repl_method=repl_method, visual_token=visual_mask,
                                                                 corr_method="spearman")

            # Fisher-z transformation for computing the average rank correlation
            grad_of_att_rank_correlation_values = np.array(list(grad_of_att_rank_correlation_dict.values()))
            z_values = np.arctanh(grad_of_att_rank_correlation_values)
            mean_z = np.mean(z_values)
            grad_of_att_rank_corr = np.tanh(mean_z)
            grad_of_att_rank_corr = np.round(grad_of_att_rank_corr, 3)

            # Explanation mask: Attention * Input_Norm
            norm_att_rank_correlation_dict = rank_correlation(model=trained_model, loader=video_loader,
                                                              fragments_weight=norm_attn_eval.fragments_weight,
                                                              repl_method=repl_method, visual_token=visual_mask,
                                                              corr_method="spearman")

            # Fisher-z transformation for computing the average rank correlation
            norm_att_rank_correlation_values = np.array(list(norm_att_rank_correlation_dict.values()))
            z_values = np.arctanh(norm_att_rank_correlation_values)
            mean_z = np.mean(z_values)
            norm_att_rank_corr = np.tanh(mean_z)
            norm_att_rank_corr = np.round(norm_att_rank_corr, 3)

            # Explanation mask: Attention * Input_Norm * Gradient
            norm_grad_att_rank_correlation_dict = rank_correlation(model=trained_model, loader=video_loader,
                                                                   fragments_weight=norm_grad_attn_eval.fragments_weight,
                                                                   repl_method=repl_method, visual_token=visual_mask,
                                                                   corr_method="spearman")

            # Fisher-z transformation for computing the average rank correlation
            norm_grad_att_rank_correlation_values = np.array(list(norm_grad_att_rank_correlation_dict.values()))
            z_values = np.arctanh(norm_grad_att_rank_correlation_values)
            mean_z = np.mean(z_values)
            norm_grad_att_rank_corr = np.tanh(mean_z)
            norm_grad_att_rank_corr = np.round(norm_grad_att_rank_corr, 3)

        att_disc_minus_all.append(att_disc_minus)
        norm_att_disc_minus_all.append(norm_att_disc_minus)
        grad_att_disc_minus_all.append(grad_att_disc_minus)
        grad_of_att_disc_minus_all.append(grad_of_att_disc_minus)
        norm_grad_att_disc_minus_all.append(norm_grad_att_disc_minus)

        att_disc_plus_all.append(att_disc_plus)
        norm_att_disc_plus_all.append(norm_att_disc_plus)
        grad_att_disc_plus_all.append(grad_att_disc_plus)
        grad_of_att_disc_plus_all.append(grad_of_att_disc_plus)
        norm_grad_att_disc_plus_all.append(norm_grad_att_disc_plus)

        att_sanity_test_all.append(att_sanity_test)
        norm_att_sanity_test_all.append(norm_att_sanity_test)
        grad_att_sanity_test_all.append(grad_att_sanity_test)
        grad_of_att_sanity_test_all.append(grad_of_att_sanity_test)
        norm_grad_att_sanity_test_all.append(norm_grad_att_sanity_test)

        if repl_fragments == "single":
            att_rank_corr_all.append(att_rank_corr)
            norm_att_rank_corr_all.append(norm_att_rank_corr)
            grad_att_rank_corr_all.append(grad_att_rank_corr)
            grad_of_att_rank_corr_all.append(grad_of_att_rank_corr)
            norm_grad_att_rank_corr_all.append(norm_grad_att_rank_corr)

        # Print the evaluation results for each split
        print("Results for split: " + str(split_id))
        table = PrettyTable()
        table.field_names = ["Measure", "Raw Attention", "Input Norm Attention", "Grad Attention", "Grad of Attention",
                             "Input Norm Grad Attention"]
        table.add_row(["Discoverability-", att_disc_minus, norm_att_disc_minus, grad_att_disc_minus,
                       grad_of_att_disc_minus, norm_grad_att_disc_minus])
        table.add_row(["Discoverability+", att_disc_plus, norm_att_disc_plus, grad_att_disc_plus,
                       grad_of_att_disc_plus, norm_grad_att_disc_plus])
        table.add_row(
            ["Sanity Violation", att_sanity_test, norm_att_sanity_test, grad_att_sanity_test, grad_of_att_sanity_test,
             norm_grad_att_sanity_test])
        if repl_fragments == "single":
            table.add_row(["Rank Correlation", att_rank_corr, norm_att_rank_corr, grad_att_rank_corr, grad_of_att_rank_corr,
                           norm_grad_att_rank_corr])
        print(table)

    # Compute overall results (averaged values across splits)
    att_disc_minus_overall = np.round(np.array(att_disc_minus_all).mean(), 3)
    norm_att_disc_minus_overall = np.round(np.array(norm_att_disc_minus_all).mean(), 3)
    grad_att_disc_minus_overall = np.round(np.array(grad_att_disc_minus_all).mean(), 3)
    grad_of_att_disc_minus_overall = np.round(np.array(grad_of_att_disc_minus_all).mean(), 3)
    norm_grad_att_disc_minus_overall = np.round(np.array(norm_grad_att_disc_minus_all).mean(), 3)

    att_disc_plus_overall = np.round(np.array(att_disc_plus_all).mean(), 3)
    norm_att_disc_plus_overall = np.round(np.array(norm_att_disc_plus_all).mean(), 3)
    grad_att_disc_plus_overall = np.round(np.array(grad_att_disc_plus_all).mean(), 3)
    grad_of_att_disc_plus_overall = np.round(np.array(grad_of_att_disc_plus_all).mean(), 3)
    norm_grad_att_disc_plus_overall = np.round(np.array(norm_grad_att_disc_plus_all).mean(), 3)

    att_sanity_test_overall = np.round(np.array(att_sanity_test_all).mean(), 3)
    norm_att_sanity_test_overall = np.round(np.array(norm_att_sanity_test_all).mean(), 3)
    grad_att_sanity_test_overall = np.round(np.array(grad_att_sanity_test_all).mean(), 3)
    grad_of_att_sanity_test_overall = np.round(np.array(grad_of_att_sanity_test_all).mean(), 3)
    norm_grad_att_sanity_test_overall = np.round(np.array(norm_grad_att_sanity_test_all).mean(), 3)

    if repl_fragments == "single":
        z_values = np.arctanh(np.array(att_rank_corr_all))
        mean_z = np.mean(z_values)
        att_rank_corr_all_mean = np.tanh(mean_z)
        att_rank_corr_overall = np.round(att_rank_corr_all_mean, 3)

        z_values = np.arctanh(np.array(norm_att_rank_corr_all))
        mean_z = np.mean(z_values)
        norm_att_rank_corr_all_mean = np.tanh(mean_z)
        norm_att_rank_corr_overall = np.round(norm_att_rank_corr_all_mean, 3)

        z_values = np.arctanh(np.array(grad_att_rank_corr_all))
        mean_z = np.mean(z_values)
        grad_att_rank_corr_all_mean = np.tanh(mean_z)
        grad_att_rank_corr_overall = np.round(grad_att_rank_corr_all_mean, 3)

        z_values = np.arctanh(np.array(grad_of_att_rank_corr_all))
        mean_z = np.mean(z_values)
        grad_of_att_rank_corr_all_mean = np.tanh(mean_z)
        grad_of_att_rank_corr_overall = np.round(grad_of_att_rank_corr_all_mean, 3)

        z_values = np.arctanh(np.array(norm_grad_att_rank_corr_all))
        mean_z = np.mean(z_values)
        norm_grad_att_rank_corr_all_mean = np.tanh(mean_z)
        norm_grad_att_rank_corr_overall = np.round(norm_grad_att_rank_corr_all_mean, 3)

    # Print the overall evaluation results
    print("Overall results (for all splits)")
    table = PrettyTable()
    table.field_names = ["Measure", "Raw Attention", "Input Norm Attention", "Grad Attention", "Grad of Attention",
                         "Input Norm Grad Attention"]
    table.add_row(["Discoverability-", att_disc_minus_overall, norm_att_disc_minus_overall, grad_att_disc_minus_overall,
                   grad_of_att_disc_minus_overall, norm_grad_att_disc_minus_overall])
    table.add_row(["Discoverability+", att_disc_plus_overall, norm_att_disc_plus_overall, grad_att_disc_plus_overall,
                   grad_of_att_disc_plus_overall, norm_grad_att_disc_plus_overall])
    table.add_row(
        ["Sanity Violation", att_sanity_test_overall, norm_att_sanity_test_overall, grad_att_sanity_test_overall,
         grad_of_att_sanity_test_overall, norm_grad_att_sanity_test_overall])
    if repl_fragments == "single":
        table.add_row(
            ["Rank Correlation", att_rank_corr_overall, norm_att_rank_corr_overall, grad_att_rank_corr_overall,
             grad_of_att_rank_corr_overall, norm_grad_att_rank_corr_overall])
    print(table)
