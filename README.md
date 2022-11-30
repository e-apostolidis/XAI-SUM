# Explaining video summarization based on the focus of attention

## PyTorch Implementation of our Attention-based Method for Explainable Video Summarization
- From **"Explaining video summarization based on the focus of attention"**, Proc. of the IEEE Int. Symposium on Multimedia (ISM), Dec. 2022.
- Written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras
- This software can be used for...

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.3` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11600K` CPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cuDNN Version` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|
3.8(.8) | 1.7.1 | 11.0 | 8005 | 1.20.2 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the [data](data) folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
```Text
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth importance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of sub-sampled frames in original video
    /n_steps                  number of sub-sampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
```
Original videos and annotations for each dataset are also available in the dataset providers' webpages: 
- <a href="https://github.com/yalesong/tvsum" target="_blank"><img align="center" src="https://img.shields.io/badge/Dataset-TVSum-green"/></a> <a href="https://gyglim.github.io/me/vsum/index.html#benchmark" target="_blank"><img align="center" src="https://img.shields.io/badge/Dataset-SumMe-blue"/></a>
</div>

## Configurations
<div align="justify">

Setup for the experimental evaluation:
 - In [`main.py`](model/main.py#L7), define the directory where the analysis results will be saved to. </div>
 - In [`data_loader.py`](model/data_loader.py#L19:L21), specify the path to the h5 file of the used dataset, and the path to the JSON file containing data about the utilized data splits.
   
Arguments in [`configs.py`](model/configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'SumMe' | 'SumMe', 'TVSum'
`--input_size` | Size of the input feature vectors. | 1024 | int > 0
`--block_size` | Size of the blocks utilized inside the attention matrix. | 60 | 0 < int ≤ 60
`--init_type` | Weight initialization method. | 'xavier' | None, 'xavier', 'normal', 'kaiming', 'orthogonal'
`--init_gain` | Scaling factor for the initialization methods. | √2 | None, float
`--n_epochs` | Number of training epochs. | 400 | int > 0
`--batch_size` | Size of the training batch, 20 for 'SumMe' and 40 for 'TVSum'. | 20 | 0 < int ≤ len(Dataset)
`--seed` | Chosen number for generating reproducible random numbers. | 12345 | None, int
`--clip` | Gradient norm clipping parameter. | 5 | float 
`--lr` | Value of the adopted learning rate. | 5e-4 | float
`--l2_req` | Value of the weight regularization factor. | 1e-5 | float
`--reg_factor` | Value of the length regularization factor. | 0.6 | 0 < float ≤ 1
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 4
