<p align='center'>  
  <img src='./examples/introduction.png'/>
</p>

# PHDNet: Painterly Image Harmonization in Dual Domains
[Paper]() 
### Introduction
**Painterly image harmonization** aims to adjust the foreground style of the painterly composite image to make it compatible with the background. A painterly composite image contains a photographic foreground object and a painterly background image.

## Datesets
Paniterly image harmonization requires two types of images: photographic image and painterly image. We cut a certain object from a photographic image by the corresponding instance mask, and then paste it onto a painterly image, generating a composite image. 
### Photographic image
We apply images from [COCO](https://arxiv.org/pdf/1405.0312.pdf) to produce the foreground objects. For each image, We select the object with foreground ratio in [0.05, 0.3] and generate the forefround mask. The selected foreground masks are provided in [selected_masks](https://pan.baidu.com/s/1x4BIPtOP02I8rcpNUZeSKA) (access code: ww1t). The training set can be downloaded from [COCO_train](http://images.cocodataset.org/zips/train2014.zip) and the test set from [COCO_test](http://images.cocodataset.org/zips/val2014.zip).
### Painterly image
We apply images from [WikiArt](https://www.wikiart.org/) to be the backgrounds. The dataset can be downloaded from [WikiArt](https://pan.baidu.com/s/192pGtJeMzj5VqTDjH6DUXg) (access code: sc0c). The training/test data list are provided in [wikiart_split](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset/Style).

The example dataset dirs:
```
your_dir
│
└───MS-COCO
│   └───SegmentationClass_select
│   │   │   XXX.png
│   │   │  ...
│   │   
│   └───train2014
│   │   │   XXX.jpg
│   │   │  ...
│   │   
│   └───val2014
│       │   XXX.jpg
│       │  ...
│   
└───wikiart
    └───WikiArt_Split
    │   │   style_class.txt
    │   │   style_train.csv
    │   │   style_val.csv
    │       
    └───unzipped_subfolders
```

## Prerequisites
- Linux
- Python 3
- PyTorch 1.10
- NVIDIA GPU + CUDA

## Getting Started
### Installation
- Clone this repo:
```bash
git clone 
cd 
```
- Prepare the [datasets](#datasets)
- Install PyTorch and dependencies from http://pytorch.org
- Install python requirements:
```bash
pip install -r requirements.txt
```

## Other Resources
- [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
- [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
