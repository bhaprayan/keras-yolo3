from keras.models import load_model
import os
from keras.layers import Input
from train import get_classes, get_anchors
from yolo3.utils import get_random_data, letterbox_image
from yolo3.model import preprocess_true_boxes, yolo_loss, yolo_body
import numpy as np
from train import create_model, create_locloss_model
from keras import backend as K
from PIL import Image
from keras.optimizers import Adam

model_path = "logs/000/ep009-loss30.814-val_loss30.951.h5"
classes_path = 'model_data/classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)

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
batch_data = []
image_data.append(image)
box_data.append(box)
batch_data.append(annotation_lines[0])

image_data = np.array(image_data)
box_data = np.array(box_data)

y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, batch_data)

# model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=False)

model = create_locloss_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=False)

# K.clear_session()

# image_input = Input(shape=(None, None, 3))

# model = yolo_body(image_input, num_anchors//3, num_classes)

# model.load_weights(model_path, by_name=True, skip_mismatch=True)


# model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

sess = K.get_session()
model_image_size = (416,416)

img = "data/5cc3a5ef4e436f43f7b5615f/images/0.jpeg"
image = Image.open(img)
boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)
# input_image_shape = K.placeholder(shape=(2, ))

# retrieve_layers = [-4,-3,-2]
# i, j, k = sess.run([model.layers[i].output for i in retrieve_layers], feed_dict={i:d for i, d in zip(model.input, [image_data, *y_true])})

# i = sess.run(model.output, feed_dict={i:d for i, d in zip(model.input, [image_data, *y_true])}) 

i = sess.run(model.output, feed_dict={model.input: image_data})


print(i)
# outputs = [i, j, k]

# total_loss = yolo_loss([*outputs, *y_true], anchors, num_classes, ignore_thresh=0.5)
