# FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator

This repository contains the reference code and dataset for the paper:

**[FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator](https://arxiv.org/abs/2508.21040)**  
Huynh Tong Dang Khoa, Dang Hoai Nam, Vo Nguyen Le Duy

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/architecture.png?raw=true#gh-dark-mode-only)

## Installation

```console
conda create --name fwgan python=3.10
conda activate fwgan
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/DAIR-Group/FW-GAN.git && cd FW-GAN
pip install -r requirements.txt
```

From [this link](https://pixeldrain.com/l/t1jhhxS1) you have to download the files `train.hdf5` and `test.hdf5` and place them into the `data` folder. You can also download our pretrained model `FW-GAN.pth` and place it under `/data/weights/FW-GAN.pth` for evaluation.

## Training

```console
python train.py --config ./configs/fw_gan_iam.yml
```


## Generate Handwtitten Text Images

To generate all samples for FID evaluation you can use the following script:

```console
python generate.py --config ./configs/fw_gan_iam.yml
```

## Handwriting synthesis and reconstruction results on IAM dataset

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_gen.png?raw=true#gh-dark-mode-only)

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_reconstruction.png?raw=true#gh-dark-mode-only)

## Handwriting synthesis on HANDS-VNOnDB dataset
![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_Vietnamese.png?raw=true#gh-dark-mode-only)

### Implementation details
This work is partially based on the code released for [HiGAN](https://github.com/ganji15/HiGAN)

## Citation
If you find this work useful, please cite our paper:

```bibtex
@misc{khoa2025fwganfrequencydrivenhandwritingsynthesis,
      title={FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator}, 
      author={Huynh Tong Dang Khoa and Dang Hoai Nam and Vo Nguyen Le Duy},
      year={2025},
      eprint={2508.21040},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.21040}, 
}
