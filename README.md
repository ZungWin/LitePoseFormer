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
