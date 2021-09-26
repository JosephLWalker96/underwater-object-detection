# RandAugment with Bounding Box Transformation

RandAugment with bounding box transformation is a Python library for dealing with image transformation. It's an unofficial reimplimentation of the RandAugment, and most of the code are adopted from [pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment), with ideas on bounding box transformation inspired by [DataAugmentationForObjectDetection](https://github.com/Paperspace/DataAugmentationForObjectDetection).

## Installation

Use the [git clone command](https://pip.pypa.io/en/stable/) to clone the project repo and use RandAugment with Bounding Box Transformation.

```bash
git clone https://github.com/JosephLWalker96/underwater-object-detection
```
After cloning the repo, set the PATH variable in your .py file and import the RandAugment module.

## Usage

After the installation, set the PATH variable and import the module and necessary dependencies.

```python
# Add the repo to PATH
import sys
sys.path.append('REPO_ROOT_DIR/underwater-object-detection')

# Import RandAugment Module
import RandAugment

# Import necessary dependencies
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
```

To start, initialize the RandAugment object containing N and M, signifying the number of transformations and the transformation magnitude. 
```python
# Number of transformations
N = 5
# Transformation magnitude
M = 30
randaug = RandAugment.augmentations.RandAugment(N, M)
```

Read in the image and bounding box
```python
# Read in the image
im = Image.open("SAMPLE_IMAGE_DIR")

# Read in the labels
labels = pd.read_csv("SAMPLE_LABELS_DIR")
label = labels.loc[labels[5] == "SAMPE_IMG_NAME"].values[0]
bbox = np.array(label[1:5])
```

Show the input image with bounding box

```python
# Calculate the two points of the bounding box
x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

# Draw out the bounding box
img_with_bbox = cv2.rectangle(np.array(im), (x1, y1), (x2, y2), (255,0,0), 5)

plt.imshow(img_with_bbox)
```
![original](images/original.png)

## Output

To get the result, call the RandAugment object in the following format. Pass in the PIL object and bounding box for the algorithm to apply the RandAugment to. Bounding box should be in the format of Python list-like object of four integers: [ X ,  Y ,  Width ,  Height ]

```python
# Call the RandAugment object to conduct transform
res = randaug(im, bbox)

# Transformed image and bounding box
transformed_im = res[0]
transformed_bbox = res[1]
```

Show the sample transformed image with bounding box

```python
# Transformed two points of the bounding box
x1, y1, x2, y2 = transformed_bbox[0], transformed_bbox[1], transformed_bbox[0] + transformed_bbox[2], transformed_bbox[1] + transformed_bbox[3]

# Draw out the bounding box
transformed_img_with_bbox = cv2.rectangle(np.array(transformed_im), (x1, y1), (x2, y2), (255,0,0), 5)

plt.imshow(transformed_img_with_bbox)
```
![transformed](images/transformed.png)

## License
[MIT](https://choosealicense.com/licenses/mit/)
