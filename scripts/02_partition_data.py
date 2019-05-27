"""Load training data and partition for training."""

import json

import cv2
import numpy as np
from toolz import pipe
from toolz.curried import map
from functools import partial

# %% load data
with open("data/labels.json") as f:
    labels = json.loads(f.read())

# images
image_ids = list(labels.keys())
images = pipe(image_ids,
              map(lambda x: "data/images/" + x + ".jpg"),
              map(cv2.imread),
              map(partial(cv2.resize, dsize=(256, 256))),
              list,
              np.array)

# labels
image_labels = pipe(labels.values(),
                    list,
                    np.array,
                    partial(np.expand_dims, axis=1))

assert images.shape[0] == image_labels.shape[0]

# %% partition into train and test data
# TODO

# %% save results
np.save(file="data/processed/X.npy", arr=images)
np.save(file="data/processed/y.npy", arr=image_labels)
