Training code for the CNN part of MDNet
=============

According to the ensemble analysis of ResNet in the paper, the classification module of ResNet is not optimal to use the information of different layers. To solve this problem, we propsoe to use, namely __ensemble_cnnection__, to better utilize multi-scale information from different layers. Our architecture is very similar with pre-act-ResNet (He et al. ECCV, 2016). However, with our simple modification (see code), the performance can be **substantially improved**. 

Out implementation is built upon the code of [Wide ResNet](https://github.com/szagoruyko/wide-residual-networks). Different from it, we found bottleneck residual block performs much better than ``tubby-like`` residual block used by WRN. More explanations are in the paper. 

## Dataset
The data is processed using the standard mean/std normalization.

Download the data to datasets/ with the following links:

[CIFAR10](https://drive.google.com/open?id=0B5jD1zn1cw8oOWw0bC10eXdxVnM)

[CIFAR100](https://drive.google.com/open?id=0B5jD1zn1cw8oWkhTbm1FUjdEdWs)


## Training
Example for CIFAR100:
```bash
model=resnet-pre-act-my-widen depth=56 widen_factor=12 dataset='cifar100' sh scripts/train_cifar.sh trial1
```
Example for CIFAR10:
```bash
model=resnet-pre-act-my-widen depth=56 widen_factor=12 dataset='cifar10' sh scripts/train_cifar.sh trial1
```
The logs are in logs/[your defined run name].
The results should be higher or lower based on different runs than the paper reported.

## Results
```
cd notebooks
jupyter notebook
```
Then go to visualize.ipynb for details. The results of pre-trained models for cifar10 and cifar100 are illustrated.


## Reference
Please cite this paper 
```bash
@inproceedings{zhang2017mdnet,
  title={MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network},
  author={Zhang, Zizhao and Xie, Yuanpu and Xing, Fuyong and Mcgough, Mason and Yang, Lin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017} 
}
```