# FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator

This repository contains the reference code and dataset for the paper FW-GAN: Frequency-Driven Handwriting Synthesis with Wave-Modulated MLP Generator.

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/architecture.png?raw=true#gh-dark-mode-only)

From [this folder](https://pixeldrain.com/l/t1jhhxS1) you have to download the files `train.hdf5` and `test.hdf5` and place them into the `data` folder. You can also download our pretrained model `FW-GAN.pth` and place it under `/data/weights/FW-GAN.pth` for evaluation.

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

![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs\Visualization_reconstruction.png?raw=true#gh-dark-mode-only)

## Handwriting synthesis on HANDS-VNOnDB dataset
![test](https://github.com/DAIR-Group/FW-GAN/blob/main/docs/Visualization_Vietnamese.png?raw=true#gh-dark-mode-only)

### Implementation details
This work is partially based on the code released for [HiGAN](https://github.com/ganji15/HiGAN)
