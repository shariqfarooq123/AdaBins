# AdaBins
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adabins-depth-estimation-using-adaptive-bins/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=adabins-depth-estimation-using-adaptive-bins) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adabins-depth-estimation-using-adaptive-bins/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=adabins-depth-estimation-using-adaptive-bins)

Official implementation of [Adabins: Depth Estimation using adaptive bins](https://arxiv.org/abs/2011.14141)
## Download links
* You can download the pretrained models "AdaBins_nyu.pt" and "AdaBins_kitti.pt" from [here](https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing)
* You can download the predicted depths in 16-bit format for NYU-Depth-v2 official test set and KITTI Eigen split test set [here](https://drive.google.com/drive/folders/1b3nfm8lqrvUjtYGmsqA5gptNQ8vPlzzS?usp=sharing)

## Colab demo 

<p>
<a href="https://colab.research.google.com/drive/1oxHflMh6eAJS7BhvP1amHvuBSirlS5Vl?usp=sharing" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

## Inference
Move the downloaded weights to a directory of your choice (we will use "./pretrained/" here). You can then use the pretrained models like so:

```python
from models import UnetAdaptiveBins
import model_io
from PIL import Image

MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
MAX_DEPTH_KITTI = 80

N_BINS = 256 

# NYU
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
pretrained_path = "./pretrained/AdaBins_nyu.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)

# KITTI
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
pretrained_path = "./pretrained/AdaBins_kitti.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

bin_edges, predicted_depth = model(example_rgb_batch)
```
Note that the model returns bin-edges (instead of bin-centers).

**Recommended way:** `InferenceHelper` class in `infer.py` provides an easy interface for inference and handles various types of inputs (with any prepocessing required). It uses Test-Time-Augmentation (H-Flips) and also calculates bin-centers for you:
```python
from infer import InferenceHelper

infer_helper = InferenceHelper(dataset='nyu')

# predict depth of a batched rgb tensor
example_rgb_batch = ...  
bin_centers, predicted_depth = infer_helper.predict(example_rgb_batch)

# predict depth of a single pillow image
img = Image.open("test_imgs/classroom__rgb_00283.jpg")  # any rgb pillow image
bin_centers, predicted_depth = infer_helper.predict_pil(img)

# predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
infer_helper.predict_dir("/path/to/input/dir/containing_only_images/", "path/to/output/dir/")

```
## TODO:
* Add instructions for Evaluation and Training.
* Add UI demo
* Remove unnecessary dependencies
