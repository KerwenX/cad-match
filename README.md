## Train on ULIP-2 model

## Install environments
```conda create -n cad python=3.8```\
```conda activate cad```\
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```

install submodule

```shell
# install PointNeXt
cd ./models/pointnext/PointNeXt
bash update.sh
bash install.sh
cd ../../

# install pointnet2
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt

# install KNN_CUDA
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
cd ../..
```


## Download datasets and initialize models
Download the used datasets and initialize models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research).

The `data/` folder should have the following structure:
```
data/
├── initialize_models
│   ├── point_bert_pretrained.pt
│   └── slip_base_100ep.pt
├── modelnet40_normal_resampled
│   ├── modelnet10_test_1024pts.dat
│   ├── modelnet10_test_1024pts_fps.dat
│   ├── modelnet10_train_1024pts.dat
│   ├── modelnet10_train_1024pts_fps.dat
│   ├── modelnet40_shape_names_modified.txt
│   ├── modelnet40_shape_names.txt
│   ├── modelnet40_test_1024pts.dat
│   ├── modelnet40_test_1024pts_fps.dat
│   ├── modelnet40_test_8192pts_fps.dat
│   ├── modelnet40_test.txt
│   ├── modelnet40_train_1024pts.dat
│   ├── modelnet40_train_1024pts_fps.dat
│   ├── modelnet40_train_8192pts_fps.dat
│   └── modelnet40_train.txt
└──ROCA
    ├── rendered_images
    ├── roca_pc
    ├── taxonomy.json
    ├── test.txt 
    └── train.txt
```

### Train model

Pretrain PointBert

If you have multiple GPUs

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main.py --model ULIP_PointBERT --npoints 8192 --lr 3e-3 --output-dir ./outputs/reproduce_pointbert_8kpts
```

Only one GPU

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model ULIP_PointBERT --npoints 8192 --lr 3e-3 --output-dir ./outputs/reproduce_pointbert_8kpts
```

### Test model

```shell
bash scripts/test_pointbert.sh <check_point_path>

# e.g.
bash scripts/test_pointbert.sh ./outputs/reproduce_pointbert_8kpts/checkpoint_best.pt
```

# Pre-trained models for zero-shot classification
Zero-shot classification on ModelNet40, 8k points pre-train, 8k points test, best checkpoint:

| model                                                                                                                                                                   | top1 | top5 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [Pointnet2(ssg)](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnet2_ssg.pt?authuser=0) | 57.7 | 78.9 |
| [PointMLP](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointmlp.pt?authuser=0)            | 60.0 | 79.4 |
| [PointBERT](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt?authuser=0)          | 60.3 | 84.0 |
| [PointNeXt](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnext.pt?authuser=0)          | 56.2 | 77.0 |
| [PointBERT_ULIP-2](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt) | 75.6 | 93.7 |



