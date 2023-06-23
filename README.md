## we use ULIP-2 model

## Install environments
```conda create -n cad python=3.8```\
```conda activate cad```\
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```


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



