# <div align="center">WeedsDetectNet</div>
Accurate weed detection is crucial for precision agriculture and sustainable crop protection. Existing methods often struggle to address high recognition errors, especially in complex environments where weeds have similar colors and shapes. This paper introduces an improved YOLOv5n-based weed detection model, WeedsDetectNet. WeedsDetectNet incorporates a green attention module to enhance the focus on green weed regions while suppressing non-target areas. An adaptive joint feature fusion method is proposed, combining low-level details such as weed color and texture with high-level semantic information, enabling better extraction of weed-specific features. Furthermore, a decoupled head design utilizes dynamic attention to separately handle classification and localization tasks, improving detection accuracy. The model was evaluated on the CottonWeedDet12, 4WEED DATASET, and a self-constructed dataset. Experimental results demonstrate that WeedsDetectNet outperforms existing methods, achieving higher average precision, lower misdetection rates, and more accurate bounding box regression. This lightweight model exhibits strong generalization ability and robustness, making it well-suited for real-world weed detection tasks. 


## <div align="center">Documentation</div>

Please refer to the quick start installation and usage example below.

### Configuration
GPU：NVIDIA A40     48GB HBM2 
CPU：Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz
cuda：12.1

### Install

Pip install the [**Python==3.8**](https://www.python.org/) environment with [**PyTorch==2.2.2**](https://pytorch.org/get-started/locally/).

```bash
conda create -n weedsdetectnet python=3.8 -y
```

```bash
conda/source activate weedsdetectnet
```

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -e .
```

```bash
pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 
```

```bash
pip install -U "openmim==0.3.9"
```

```bash
mim install "mmengine==0.10.4"
```
```bash
mim install "mmcv==2.2.0"
```

### Datasets
Self-built dataset：[https://drive.google.com/drive/folders/1JIuWTEHDsO3MxIktByN3Qn2xBXhHsdyz?usp=sharing](https://drive.google.com/drive/folders/1JIuWTEHDsO3MxIktByN3Qn2xBXhHsdyz?usp=sharing)

CottonWeedDet12 dataset：[https://zenodo.org/records/7535814](https://zenodo.org/records/7535814)

4WEED DATASET dataset：[https://osf.io/w9v3j/](https://osf.io/w9v3j/)

Divide the dataset：training set: validation set: test set=7:2:1

Self-built dataset（ 6 categories and 1990 images）：training set: validation set: test set=1390：396：204

CottonWeedDet12 dataset（ 12 categories and 5648 images）:training set: validation set: test set=3953：1129：566

4WEED DATASET dataset（ 4 categories and 619 images）：training set: validation set: test set=432：123：64

Attention: After downloading the self built dataset, modify the paths in the train.txt, val.txt, and test.txt files before use.

### Python

The following is the training and testing process:

```python

# train
CUDA_VISIBLE_DEVICES=5 python train.py  # default：cache=True,imgsz=640,epochs=500,batch=8
```
```python

# test
CUDA_VISIBLE_DEVICES=5 python test.py  # default：cache=True,imgsz=640,epochs=500,batch=8
```


