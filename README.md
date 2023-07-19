# SplitBeam: Effective and Efficient Beamforming in Wi-Fi Networks Through Split Computing

This is the official repository of our paper "SplitBeam: Effective and Efficient Beamforming in Wi-Fi Networks Through Split Computing" presented at ICDCS 2023.

## Datasets

The datasets used in the paper are hosted at [Hugging Face Datasets](https://huggingface.co/datasets/yoshitomo-matsubara/mu-mimo).  
You can use the following commands to download the datasets:
```shell
mkdir -p ./resource/datasets/
git lfs install
git clone https://huggingface.co/datasets/yoshitomo-matsubara/mu-mimo ./resource/datasets/
```

## Important Notice
If you have any questions regarding MATLAB and/or datasets, please directly contact 
[`Niloofar Bahadori`](https://niloobahadori.github.io/) as she provided MATLAB code and Python code to wrap 
the MATLAB code, set up the MATLAB environment, and created the datasets.


## Setup
- Python 3.8
- MATLAB 2021R
- conda

```shell
conda env create -f environment.yml
~/anaconda3/bin/pip3 install -r requirements.txt --user
```


## Run

### Train models

Use scripts under `scripts/`  
e.g.,  
```shell
sh scripts/2x2-20mhz/env1-batch.sh
sh scripts/2x2-20mhz/env2-batch.sh
```

### Test models with post-training quantization

Use scripts under `scripts-quantization/`  
e.g.,  
```shell
sh scripts-quantization/2x2-20mhz/env1-batch.sh
sh scripts-quantization/2x2-20mhz/env2-batch.sh
```


## Citation

```bibtex
@inproceedings{bahadori2023splitbeam,
  title={{SplitBeam: Effective and Efficient Beamforming in Wi-Fi Networks Through Split Computing}},
  author={Bahadori, Niloofar and Matsubara, Yoshitomo and Levorato, Marco and Restuccia, Francesco},
  booktitle={2023 IEEE 43rd International Conference on Distributed Computing Systems (ICDCS)},
  pages={864--874},
  year={2023},
  organization={IEEE}
}
```
