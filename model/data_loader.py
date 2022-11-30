# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import json
from typing import Tuple


class VideoData(Dataset):
    def __init__(self, video_type, split_index, step=20) -> None:
        """ Custom Dataset class wrapper for loading the frame features and other video characteristics.

        :param str video_type: The Dataset being used, SumMe or TVSum
        :param int split_index: The index of the data-split being used
        :param int step: The step size (in frames) in the case of uniform video segmentation (provided for 2 fps)
        """
        self.name = video_type.lower()
        self.filename = f".../XAI-SUM/data/{video_type}/eccv16_dataset_{self.name}_google_pool5.h5"
        self.splits_filename = f".../XAI-SUM/data/splits/{self.name}_splits.json"
        self.split_index = split_index
        self.step = step * 15  # to account for 2 fps <-> 30 fps changes

        hdf = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_fragments = [], []

        with open(self.splits_filename) as f:
            data = json.loads(f.read())
            self.split = data[split_index]

        for video_name in self.split['test_keys']:
            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
            if self.step != 0:
                n_frames = np.array(hdf[f"{video_name}/n_frames"])
                iters = int(np.ceil(n_frames / self.step))
                change_points = np.array([[count * self.step, (count + 1) * self.step] for count in range(iters)])
                change_points = torch.from_numpy(change_points)
            else:
                raise NotImplementedError("Choose a valid number (sampled frames) for the length of the video fragment")

            self.list_frame_features.append(frame_features)
            self.list_fragments.append(change_points)

        hdf.close()

    def __len__(self) -> int:
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        return len(self.list_frame_features)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """ Function to be called for the index operator of `VideoData` Dataset

        :param int index: The above-mentioned id of the data
        :return: A tuple containing the frame features, fragments and the name, of the selected video
        """
        video_name = self.split['test_keys'][index]
        frame_features = self.list_frame_features[index]
        fragments = self.list_fragments[index]

        return frame_features, fragments, video_name


def get_loader(video_type, split_index, step=20) -> VideoData:
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.

    :param str video_type: The Dataset being used, SumMe or TVSum
    :param int split_index: The index of the Dataset split being used
    :param int step: The step size (in frames) for uniform video segmentation
    :return: The Dataset in use
    """
    return VideoData(video_type, split_index, step)


if __name__ == '__main__':
    pass
