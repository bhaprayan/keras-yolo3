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
import ipdb
import time
from filter_loss import filter_high_loss, filter_low_loss
from yolo import YOLO
import matplotlib.pyplot as plt
import json

model_path = "logs/000/ep009-loss30.814-val_loss30.951.h5"
classes_path = 'model_data/classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)
input_shape = (416,416)

model = create_locloss_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=True)
sess = K.get_session()

annotation_path = 'updated_train_nuro.txt'
uuid_path = 'updated_uuid_nuro.txt'
val_split = 0.99
with open(annotation_path) as f:
    annotation_lines = f.readlines()
with open(uuid_path) as f:
    uuid_lines = f.readlines()
num_val = int(len(annotation_lines)*val_split)
num_train = len(annotation_lines) - num_val

annotate_dict = {}
for i, line in enumerate(annotation_lines):
    annotate_dict[line.split()[0]] = i

# high_loss_idx = filter_high_loss(10)
# high_loss_idx = filter_low_loss(10)
# extract only top 100 entries for now
# n = len(high_loss_idx)
n = len(annotation_lines)
start = time.time()
for i in range(10):
    # idx = annotate_dict[high_loss_idx[i]] # extract line number of high loss image from dict
    annotation_line = annotation_lines[i] # extract line text
    image, box = get_random_data(annotation_line, input_shape, random=False)
    # extract image location
    image_data = []
    box_data = []
    batch_data = []
    image_data.append(image)
    box_data.append(box)
    batch_data.append(annotation_line)
    uuid_data = uuid_lines[i].split()

    image_data = np.array(image_data)
    box_data = np.array(box_data)
    uuid_data = np.array(uuid_data)

    y_true, obj_uuid = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, batch_data, uuid_data)

    tensor_map = {}

    for j in range(len(obj_uuid)):
        # TODO: retrieve uuid scale mapping
        flat_tensor = obj_uuid[j].flatten()
        flat_tensor = flat_tensor[np.nonzero(flat_tensor)]
        tensor_map[str(j)+'_uuid'] = flat_tensor.tolist()
    
    out = sess.run(model.output, feed_dict={k:d for k, d in zip(model.input, [image_data, *y_true])})

    for grid_n in range(len(out)-1):
        # TODO: retrieve dict name mapping
        flat_tensor = out[grid_n].flatten()
        flat_tensor = flat_tensor[np.nonzero(flat_tensor)]
        tensor_map[str(grid_n)+'_grid'] = flat_tensor.tolist()

    try:
        img_name = annotation_line.split()[0]
        frame_no = img_name.split('/')[-1].split('.')[0]
        subtask = img_name.split('/')[-3]
        task = img_name.split('/')[-4]
        tensor_map['image_name'] = img_name
        tensor_map['frame_no'] = frame_no
        tensor_map['subtask'] = subtask
        tensor_map['task'] = task
        with open(str(i) + '_' + 'data.json', 'w') as fp:
            json.dump(tensor_map, fp, indent=4, sort_keys=True)
        print(img_name)
    except:
        continue

end = time.time()
print('Total time:', end-start)

# model = YOLO() 
# i = 0
# for i in range(n):
#     img = high_loss_idx[i]
#     image = Image.open(img)
#     idx = annotate_dict[high_loss_idx[i]] # extract line number of high loss image from dict
#     annotation_line = annotation_lines[idx] # extract line text
#     print(img, annotation_line)
#     box = np.array([np.array(list(map(int,box.split(',')))) for box in annotation_line.split()[1:]]) 
#     box[:,[0,1]] = box[:,[1,0]]
#     box[:,[2,3]] = box[:,[3,2]]
#     img_arr = model.detect_image_bboxes(image, box)
#     img_arr.save(str(i) + '_detect.jpeg')


    # img_arr = model.detect_image(image)
    # model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=False)

# K.clear_session()

# image_input = Input(shape=(None, None, 3))

# model = yolo_body(image_input, num_anchors//3, num_classes)

# model.load_weights(model_path, by_name=True, skip_mismatch=True)


# model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

# model_image_size = (416,416)

# img = "data/5cc3a5ef4e436f43f7b5615f/images/0.jpeg"
# image = Image.open(img)
# boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
# image_data = np.array(boxed_image, dtype='float32')
# image_data /= 255.
# image_data = np.expand_dims(image_data, 0)
# input_image_shape = K.placeholder(shape=(2, ))

# retrieve_layers = [-4,-3,-2]
# i, j, k = sess.run([model.layers[i].output for i in retrieve_layers], feed_dict={i:d for i, d in zip(model.input, [image_data, *y_true])})

# i = sess.run(*model.output, feed_dict={model.input: image_data})

# print(i)
# outputs = [i, j, k]

# total_loss = yolo_loss([*outputs, *y_true], anchors, num_classes, ignore_thresh=0.5)
