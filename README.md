# WeedsDetectNet
Accurate weed detection is crucial for precision agriculture and green crop protection. Existing methods often struggle with high recognition errors, particularly in complex environments where weeds share similar colors and shapes. This paper presents WeedsDetectNet, an improved YOLOv5n model for weed detection. WeedsDetectNet introduces a green attention module to enhance focus on green weed regions while suppressing non-target areas. An adaptive joint feature fusion method is proposed to combine low-level details such as weed color and texture with high-level semantic information, enabling better extraction of weed-specific features. Additionally, a decoupled head design utilizes dynamic attention to separately handle classification and localization tasks, improving detection accuracy. The model is evaluated on the CottonWeedDet12 and 4WEED DATASET, as well as a self-constructed dataset. Experimental results demonstrate that WeedsDetectNet outperforms existing methods, achieving higher mean average precision, lower misidentification rates, and more accurate bounding box regression. This lightweight model exhibits strong generalization and robustness, making it suitable for real-world weed detection tasks. 


## <div align="center">Documentation</div>

Please refer to the quick start installation and usage example below.

<details open>
<summary>Configuration</summary>
GPU：NVIDIA A40     48GB HBM2 
CPU：Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz
cuda：12.1

<summary>Install</summary>

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



### Python

YOLO may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```

See YOLO [Python Docs](https://docs.ultralytics.com/usage/python/) for more examples.

</details>

