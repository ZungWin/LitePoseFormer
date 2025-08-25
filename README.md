# LitePoseFormer: A Lightweight Transformer Method for 3D Human Pose Estimation

This repository is the offical Pytorch implementation of LitePoseFormer: A Lightweight Transformer Method for 3D Human Pose Estimation.

## Environment
The code is developed and tested under the following environment

- Python 3.8.2
- PyTorch 1.7.1
- CUDA 11.0
- 
## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Train the model

To train on Human3.6M:

```bash
python main.py
```

## Test the model

To test on pretrained model on Human3.6M:

```bash
python main.py --test --reload --previous_dir 'checkpoint/pretrained'
```


## Citation

If you find our work useful in your research, please consider citing:

@inproceedings{Lyu2025LP,
  title={LitePoseFormer: A Lightweight Transformer Method for 3D Human Pose Estimation},
  author={Zhangwen, Lyu and Yuhe, Zhu and Rong, Liu and Yinwei, Zhan},
  booktitle={The Visual Computer},
  year={2025}
}


## Acknowledgement

Our code is built on top of [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks) and is extended from the following repositories. We thank the authors for releasing the codes. 
- [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)
- [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch)

## Licence

This project is licensed under the terms of the MIT license.
