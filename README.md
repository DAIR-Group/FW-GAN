# FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator

This repository contains the reference code and dataset for the paper:

**[FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator](https://www.sciencedirect.com/science/article/pii/S095741742503790X)**  
Huynh Tong Dang Khoa, Dang Hoai Nam, Vo Nguyen Le Duy

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/architecture.png?raw=true#)

## Installation

```console
conda create --name fwgan python=3.10
conda activate fwgan
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/DAIR-Group/FW-GAN.git && cd FW-GAN
pip install -r requirements.txt
```

We provide our pretrained model weights and datasets on [Huggingface](https://huggingface.co/DAIR-Group/FW_GAN/tree/main). For training, please download the files the files `train.hdf5` and `test.hdf5` and place them into the `data` folder. For quick evaluation, you can also download the pretrained model `FW-GAN.pth` and place it under `/data/weights/FW-GAN.pth`.

## Training

```console
python train.py --config ./configs/fw_gan_iam.yml
```


## Generate Handwritten Text Images

To generate all samples for FID evaluation you can use the following script:

```console
python generate.py --config ./configs/fw_gan_iam.yml
```

## Handwriting synthesis and reconstruction results on IAM dataset

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_gen.png?raw=true#)

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_reconstruction.png?raw=true#)

## Handwriting synthesis on HANDS-VNOnDB dataset
![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_Vietnamese.png?raw=true#)

### Implementation details
This work is partially based on the code released for [HiGAN](https://github.com/ganji15/HiGAN)

## Citation
If you find this work useful, please cite our paper:

```bibtex
@article{TONGDANGKHOA2026130175,
title = {FW-GAN: Frequency-driven handwriting synthesis with wave-modulated MLP generator},
journal = {Expert Systems with Applications},
volume = {299},
pages = {130175},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.130175},
url = {https://www.sciencedirect.com/science/article/pii/S095741742503790X},
author = {Huynh {Tong Dang Khoa} and Dang {Hoai Nam} and Vo {Nguyen Le Duy}},
keywords = {Handwritten text synthesis, Wavelet transform, One-shot learning, Vietnamese handwriting, Synthetic data, Generative adversarial networks},
}
