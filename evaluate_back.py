from keras.models import load_model
import os
from keras.models import load_model
from yolo3.model import yolo_body
from keras.layers import Input
from train import get_classes, get_anchors
from yolo3.utils import get_random_data
from yolo3.model import preprocess_true_boxes
import numpy as np

model_path = "logs/000/ep009-loss30.814-val_loss30.951.h5"
classes_path = 'model_data/classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)

yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
yolo_model.load_weights(model_path)

annotation_path = 'train_nuro.txt'
val_split = 0.99
with open(annotation_path) as f:
    annotation_lines = f.readlines()
num_val = int(len(annotation_lines)*val_split)
num_train = len(annotation_lines) - num_val
input_shape = (416,416)

n = len(annotation_lines)
image, box = get_random_data(annotation_lines[0], input_shape, random=False)

image_data = []
box_data = []
image_data.append(image)
box_data.append(box)

image_data = np.array(image_data)
box_data = np.array(box_data)
batch_data = []
batch_data.append(annotation_lines[0])

y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, batch_data)

# model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=False)
