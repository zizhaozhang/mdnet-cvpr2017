Code for the CNN part for the paper of MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network
=============

According to the ensemble analysis of ResNet in the paper, the classification module of ResNet is not optimal to use the information of different layers. To solve this problem, we propsoe to use, namely __ensemble_cnnection__, to better utilize multi-scale information from different layers. Our architecture is most similar with pre-act-ResNet. However, with our simple modification (see code), the performance can be substantially improved. 

Out implementation is built upon the code of [Wide ResNet](https://github.com/szagoruyko/wide-residual-networks). In addition, we found bottleneck residual block performs much better than ``tubby-like`` residual block used by WRN. 


Results on CIFAR 10&100:

<img src="https://github.com/zizhaozhang/mdnet-cvpr2017/png/cifar-table.png" width="300"/>
<img src="https://github.com/zizhaozhang/mdnet-cvpr2017/png/cifar-curve.png" width="300"/>


## Dataset
```bash
sh datasets/download_cifar.sh
```

## Training
Example for cifar 100:
```bash
model=resnet-pre-act-my-widen depth=56 widen_factor=12 dataset='./datasets/cifar100_mean_std.t7' sh scripts/train_cifar.sh
```

## Reference
```bash
@inproceedings{zhang2017mdnet,
  title={MDNet: A Semantically and Visually Interpretable Medical Image Diagnosis Network},
  author={Zhang, Zizhao and Xie, Yuanpu and Xing, Fuyong and Mcgough, Mason and Yang, Lin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017} 
}
```
