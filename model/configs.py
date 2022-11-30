# -*- coding: utf-8 -*-
import argparse
import torch
import pprint


class Config(object):
    def __init__(self, **kwargs):
        """ Configuration Class: set kwargs as class attributes with setattr. """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        """ Pretty-print configurations in alphabetical order. """
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """ Get configurations as attributes of class
        1. Parse configurations with argparse.
        2. Create Config class initialized with parsed kwargs.
        3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--dataset', type=str, default='SumMe', help='Dataset to be used [SumMe / TVSum]')
    # parser.add_argument('--split_index', type=int, default=0, help='Data split to be used [0-4]')
    parser.add_argument('--replacement_method', type=str, default='slice-out', help='Applied replacement function [slice-out, input-mask, random, attention-mask]')
    parser.add_argument('--replaced_fragments', type=str, default='batch', help='Amount of replaced fragments [batch / single]')
    parser.add_argument('--visual_mask', type=str, default='black-frame', help='Type of visual mask [black-frame / white-frame]')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
