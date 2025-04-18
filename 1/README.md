<<<<<<< HEAD
# MV3Dseg
=======

# MV3Dseg

This repository is for **MV3Dseg** introduced in the following paper

## Installation

### Requirements
- pytorch >= 1.8 
- yaml
- easydict
- pyquaternion
- [lightning](https://github.com/Lightning-AI/lightning) (tested with pytorch_lightning==1.3.8 and torchmetrics==0.5)
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) (pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==2.1.16 and cuda==11.1, pip install spconv-cu111==2.1.16)
- [torchsparse](https://github.com/mit-han-lab/torchsparse) (optional for MinkowskiNet and SPVCNN. sudo apt-get install libsparsehash-dev, pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0)

## Data Preparation

### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder.
```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
        |   └── image_2/ 
        |   |   ├── 000000.png
        |   |   ├── 000001.png
        |   |   └── ...
        |   calib.txt
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org/) with lidarseg and extract it.
```
./dataset/
├── 
├── ...
└── nuscenes/
    ├──v1.0-trainval
    ├──v1.0-test
    ├──samples
    ├──sweeps
    ├──maps
    ├──lidarseg
```

## Training
### SemanticKITTI
You can run the training with
```shell script
cd <root dir of this repo>
python main.py --log_dir MV3Dseg_semkitti --config config/MV3Dseg-semantickitti.yaml --gpu 0
```
The output will be written to `logs/SemanticKITTI/MV3Dseg_semkitti` by default. 
### NuScenes
```shell script
cd <root dir of this repo>
python main.py --log_dir MV3Dseg_nusc --config config/MV3Dseg-nuscenese.yaml --gpu 0 1 2 3
```

### Vanilla Training without MV3Dseg
We take SemanticKITTI as an example.
```shell script
cd <root dir of this repo>
python main.py --log_dir baseline_semkitti --config config/MV3Dseg-semantickitti.yaml --gpu 0 --baseline_only
```

## Testing
You can run the testing with
```shell script
cd <root dir of this repo>
python main.py --config config/MV3Dseg-semantickitti.yaml --gpu 0 --test --num_vote 12 --checkpoint <dir for the pytorch checkpoint>
```
Here, `num_vote` is the number of views for the test-time-augmentation (TTA). We set this value to 12 as default (on a Tesla-V100 GPU), and if you use other GPUs with smaller memory, you can choose a smaller value. `num_vote=1` denotes there is no TTA used, and will cause about ~2\% performance drop.

## Robustness Evaluation
Please download all subsets of [SemanticKITTI-C](https://arxiv.org/pdf/2301.00970.pdf) from [this link](https://cuhko365-my.sharepoint.com/personal/218012048_link_cuhk_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F218012048%5Flink%5Fcuhk%5Fedu%5Fcn%2FDocuments%2FSemanticKITTIC&ga=1) and extract them.
```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
    ├──SemanticKITTI-C
        ├── clean_data/           
        ├── dense_16beam/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
	    ...
```
You can run the robustness evaluation with
```shell script
cd <root dir of this repo>
python robust_test.py --config config/MV3Dseg-semantickitti.yaml --gpu 0  --num_vote 12 --checkpoint <dir for the pytorch checkpoint>
```

## Acknowledgements
Code is built based on [SPVNAS](https://github.com/mit-han-lab/spvnas), [Cylinder3D](https://github.com/xinge008/Cylinder3D), [xMUDA](https://github.com/valeoai/xmuda) and [SPCONV](https://github.com/traveller59/spconv), [2DPASS](https://github.com/xinge008/Cylinder3D).

## License
This repository is released under MIT License (see LICENSE file for details).



>>>>>>> 162c4e4 (first commit)
