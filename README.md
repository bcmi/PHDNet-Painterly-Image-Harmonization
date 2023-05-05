

# PHDNet: Painterly Image Harmonization in Dual Domains

This is the official repository for the following paper:

> **Painterly Image Harmonization in Dual Domains**  [[arXiv]](http://arxiv.org/abs/2212.08846)<br>
>
> Junyan Cao, Yan Hong, Li Niu<br>
> Accepted by **AAAI 2023**.


**Painterly image harmonization** aims to adjust the foreground style of the painterly composite image to make it compatible with the background. A painterly composite image contains a photographic foreground object and a painterly background image.


<p align='center'>  
  <img src='./examples/introduction.png'  width=70% />
</p>

## Datesets
Paniterly image harmonization requires two types of images: photographic image and painterly image. We cut a certain object from a photographic image by the corresponding instance mask, and then paste it onto a painterly image, generating a composite image. 
### Photographic image
We apply images from [COCO](https://arxiv.org/pdf/1405.0312.pdf) to produce the foreground objects. For each image, We select the object with foreground ratio in [0.05, 0.3] and generate the forefround mask. The selected foreground masks are provided in [selected_masks](https://pan.baidu.com/s/1x4BIPtOP02I8rcpNUZeSKA) (access code: ww1t). The training set can be downloaded from [COCO_train](http://images.cocodataset.org/zips/train2014.zip) (alternative: [Baidu Cloud](https://pan.baidu.com/s/19d5BsbcVwIH4jMXGLpcBtg) (access code: nfsh)) and the test set from [COCO_test](http://images.cocodataset.org/zips/val2014.zip) (alternative: [Baidu Cloud](https://pan.baidu.com/s/1bOZzpoyO3aNPR3NSqAVXkw) (access code: nsvj)).
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
git clone https://github.com/bcmi/PHDNet-Painterly-Image-Harmonization.git
# cd to this repo's root dir
```

- Prepare the datasets.

- Install PyTorch and dependencies from http://pytorch.org.

- Install python requirements:

```bash
pip install -r requirements.txt
```

- Download pre-trained VGG19 from [Baidu Cloud](https://pan.baidu.com/s/1HljOE-4Q2yUeeWmteu0nNA) (access code: pc9y).

### PHDNet train/test
- Train PHDNet: 

```bash
cd PHDNet/scripts
bash train_phd.sh
```

The trained model would be saved under `./<checkpoint_dir>/<name>/`.

If you want to load a model then continue to train it, add `--continue_train` and set the `--epoch XX` in `train_phd.sh`. It would load the model `./<checkpoint_dir>/<name>/net_G_<epoch>.pth`.

Remember to modify the `content_dir` and `style_dir` to the corresponding path of each dataset in `train_phd.sh`.

- Test PHDNet:

```bash
cd PHDNet/scripts
bash test_phd.sh
```

It would load the model `./<checkpoint_dir>/<name>/net_G_<epoch>.pth` then save the visualization under `./<checkpoint_dir>/<name>/web/TestImages/`

Our pre-trained model is available on [Baidu Cloud](https://pan.baidu.com/s/1D6iAS6Sli1QggLp-E9EvyQ) (access code: po7q).

- Note: `<...>` means modifiable parameters.


### Experimental results

Our method is especially good at handling the background paintings with periodic textures and patterns, because we leverage the information from both spatial domain and frequency domain. 

<p align='center'>  
  <img src='./examples/experimental_results.jpg'  width=90% />
</p>

## Other Resources
- [Awesome-Image-Harmonization](https://github.com/bcmi/Awesome-Image-Harmonization)
- [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)
