# A Study on the Use of Attention for Explaining Video Summarization

## PyTorch implementation of the software used in:
- **"Explaining Video Summarization Based on the Focus of Attention"**, Proc. of the IEEE Int. Symposium on Multimedia (ISM), Dec. 2022. (written by Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris and Ioannis Patras)
- **"A Study on the Use of Attention for Explaining Video Summarization"**, Proc. of the NarSUM workshop at ACM Multimedia 2023 (ACM MM), Oct.-Nov. 2023. (written by Evlampios Apostolidis, Vasileios Mezaris and Ioannis Patras)
- This software can be used for studying our method for producing explanations for the outcomes of attention-based video summarization models, and re-producing the reported exprerimental results in our papers. 

## Main dependencies
Developed, checked and verified on an `Ubuntu 20.04.5` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11600K` CPU. Main packages required:
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

## Running an experiment
<div align="justify">

To run an experiment using one of the aforementioned datasets and considering all of its randomly-generated splits (stored in the JSON file included in the [data/splits](/data/splits) directory), execute the following command:

```
python model/main.py --summarization_method 'method_name' --dataset 'dataset_name' --replacement_method 'replacement_method_name' --replaced_fragments 'set_of_repl_fragments' --visual_mask 'visual_mask_name'
```
where, `method_name` refers to the name of the used video summarization method, `dataset_name` refers to the name of the used dataset, `replacement_method_name` refers to the applied replacement function in fragments of the input data, `set_of_repl_fragments` refers to the amount of replaced fragments of the input data, and `visual_mask_name` refers to the type of the used mask when replacing fragments of the input data.

After executing the above command you get the overall results for each different data split, as well as the overall results that are computed by averaging the obtained scores across data splits. Please note that, the results when fragments' replacement is based on "Randomization" might be slightly different from the reported ones, as we did not use a fixed seed value in our experiments.

## Configurations
<div align="justify">

Setup for the experimental evaluation:
 - In [`main.py`](main.py#L66:L80), specify the path to the pretrained models of the used method for video summarization. 
 - In [`data_loader.py`](data_loader.py#L19:L20), specify the paths to the h5 file of the used dataset, and the JSON file containing data about the used data splits.</div>
   
Arguments in [`configs.py`](configs.py): 
|Parameter name | Description | Default Value | Options
| :--- | :--- | :---: | :---:
`summarization method` | The used video summarization method. | 'CA-SUM' | 'CA-SUM', 'VASNet', 'SUM-GDA'
`dataset` | The used dataset. | 'SumMe' | 'SumMe', 'TVSum'
`replacement_method` | The applied replacement function. | 'slice-out' | 'slice-out', 'input-mask', 'random', 'attention-mask'
`replaced_fragments` | The amount of replaced fragments. | 'batch' | 'batch', 'single'
`visual_mask` | The used visual mask for replacement. | 'black-frame' | 'black-frame', 'white-frame'

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publications:

E. Apostolidis, V. Mezaris, I. Patras, "<b>A Study on the Use of Attention for Explaining Video Summarization</b>", Proc. NarSUM workshop at ACM Multimedia 2023 (ACM MM), Oct.-Nov. 2023.
</div>

BibTeX:

```
@INPROCEEDINGS{9666088,
    author    = {Apostolidis, Evlampios and Mezaris, Vasileios and Patras, Ioannis},
    title     = {A Study on the Use of Attention for Explaining Video Summarization},
    booktitle = {Proc. of the 2nd Workshop on User-Centric Narrative Summarization of Long Videos (NarSUM'23) at ACM Multimedia 2023},
    month     = {October},
    year      = {2023}
}
```

E. Apostolidis, G. Balaouras, V. Mezaris, I. Patras, "<b>Explaining Video Summarization Based on the Focus of Attention</b>", Proc. IEEE Int. Symposium on Multimedia (ISM), Dec. 2022.
</div>

BibTeX:

```
@INPROCEEDINGS{9666088,
    author    = {Apostolidis, Evlampios and Balaouras, Georgios and Mezaris, Vasileios and Patras, Ioannis},
    title     = {Explaining Video Summarization Based on the Focus of Attention},
    booktitle = {2022 IEEE International Symposium on Multimedia (ISM)},
    month     = {December},
    year      = {2022}
}
```

## License
<div align="justify">

Copyright (c) 2022-2023, Evlampios Apostolidis, Georgios Balaouras, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-951911 AI4Media. </div>
