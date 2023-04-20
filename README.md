# Taming Diffusion Models for Music-Conditioned Conductor Motion Generation

## Installation

Please refer to [install.md](/Diffusion_Stage/install.md) for detailed installation.

## Training

### Prepare the ConductorMotion100 dataset:

- The training set：https://pan.baidu.com/s/1Pmtr7V7-9ChJqQp04NOyZg?pwd=3209
- The validation set：https://pan.baidu.com/s/1B5JrZnFCFvI9ABkuJeWoFQ?pwd=3209 
- The test set：https://pan.baidu.com/s/18ecHYk9b4YM5YTcBNn37qQ?pwd=3209 


### Train the music encoder in Contrastive_Stage with the following command:

```shell 
python M2SNet_train.py --dataset_dir <Your Dataset Dir> 
```

### Train the diffusion model in Diffusion_Stage with the following command:

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -u tools/train.py \
    --name kit_baseline_dp_2gpu_8layers_1000 \
    --batch_size 128 \
    --times 50 \
    --num_epochs 50 \
    --dataset_name kit \
    --num_layers 8 \
    --diffusion_steps 1000 \
    --data_parallel \
    --gpu_id 0 1
```

## Evaluation and Visualization

TODO

## Acknowledgement
This repo partially uses code from [VirtualConductor](https://github.com/ChenDelong1999/VirtualConductor) and [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse).