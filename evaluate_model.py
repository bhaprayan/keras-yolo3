model_path = 'logs/000/ep009-loss30.814-val_loss30.951.h5'
from keras.models import load_model
from PIL import Image
from yolo3.utils import letterbox_image
from yolo import YOLO, detect_video
import matplotlib.pyplot as plt
import glob
annotation_path = 'train_nuro.txt'
model = YOLO()
val_split = 0.99
with open(annotation_path) as f:
    lines = f.readlines()
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
batch_size = 32
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
x = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
from train import *
x = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
annotation_path = 'train_nuro.txt'
log_dir = 'logs/000/'
classes_path = 'model_data/classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)
input_shape = (416,416) # multiple of 32, hw
x = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
data = x.__next__()
len(data)
data[0].shape
len(data[0])
data
data[0]
data[0][0].shape
data[0][1].shape
print(data[0][...].shape)
print(data[0][:].shape)
print(data[0][i].shape for i in data[0])
data[0][1].shape
model.evaluate?
model.yolo_model.evaluate?
model.yolo_model.predict(data)
data[0][0].shape
model.yolo_model.predict(data[0][0])
pred = model.yolo_model.predict(data[0][0])
pred.shape
len(pred)
pred[0].shape
pred[1].shape
pred[2].shape
model.yolo_model.evaluate(data[0][0], data[0][1:])
model_path
model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=model_path)
model.evaluate?
model.evaluate(data[0][0], data[0][:])
model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
model.evaluate(data[0][0], data[0][:])
model.evaluate(data[0][0], data[0][1:])
model.evaluate(data[0][0], *data[0][:])
data[0][0]
data[0][0].shape
data[0][:]
len(data[0][:])
model.evaluate(data[0][0], *(data[0][:]))
model.evaluate(*(data[0][:]), *(data[0][:]))
model.evaluate(*(data[0][:]))
data
data[1].shape
model.evaluate_generator?
len(lines)
batch_size
input_shape
anchors
num_classes
num_train
model.evaluate_generator?
model.evaluate_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes))
model.evaluate_generator?
model.evaluate_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes), steps=1)
%history -f evaluate_model.py
