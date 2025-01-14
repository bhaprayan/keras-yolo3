model_path = 'logs/000/ep009-loss30.814-val_loss30.951.h5'

import glob

import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image

from train import *
from yolo import YOLO, detect_video
from yolo3.utils import letterbox_image

import ipdb


annotation_path = 'train_nuro.txt'
log_dir = 'logs/000/'
classes_path = 'model_data/classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)
input_shape = (416,416) # multiple of 32, hw

# read annotations
val_split = 0.99 # keep small split for testing
with open(annotation_path) as f:
    lines = f.readlines()
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
batch_size = 1
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=model_path, grid_loss=False)

model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

# ipdb.set_trace()

model.evaluate_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes), steps=1, max_queue_size=1)
# Turns out setting queue size to n fetches data n times in the beginning 
