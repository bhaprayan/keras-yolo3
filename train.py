"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train_nuro.txt'
    log_dir = 'logs/001/'
    classes_path = 'model_data/classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/darknet53_weights.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.99
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def tf_print(op, tensors, message=None):
    def print_message(x):
        sys.stdout.write(message + " %s\n" % x)
        return x

    prints = [tf.py_func(print_message, [tensor], tensor.dtype) for tensor in tensors]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op

def model_loss_lambda(*args, **kwargs):
    return yolo_loss(*args, **kwargs)['loss']

def model_grid_loss_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['loss']

def model_grid_loss_xy_0_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['xy_loss_grid_0']

def model_grid_loss_xy_1_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['xy_loss_grid_1']

def model_grid_loss_xy_2_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['xy_loss_grid_2']

def model_grid_loss_wh_0_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['wh_loss_grid_0']

def model_grid_loss_wh_1_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['wh_loss_grid_1']

def model_grid_loss_wh_2_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['wh_loss_grid_2']

def model_grid_loss_class_0_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['class_loss_grid_0']

def model_grid_loss_class_1_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['class_loss_grid_1']

def model_grid_loss_class_2_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['class_loss_grid_2']

def model_grid_loss_confidence_0_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['confidence_loss_grid_0']

def model_grid_loss_confidence_1_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['confidence_loss_grid_1']

def model_grid_loss_confidence_2_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['confidence_loss_grid_2']

def model_output_0_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['yolo_output_0']

def model_output_1_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['yolo_output_1']

def model_output_2_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['yolo_output_2']

def model_object_mask_0_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['object_mask_0']

def model_object_mask_1_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['object_mask_1']

def model_object_mask_2_lambda(*args, **kwargs):
    dict_loss = yolo_loss(*args, **kwargs) 
    return dict_loss['object_mask_2']

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', grid_loss=False):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    # batch_details = K.placeholder(shape=(1,))

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    if grid_loss:
        model_loss = Lambda(model_grid_loss_lambda, output_shape=(1, ), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        
        model = Model([model_body.input, *y_true], model_loss)    
    else:
        model_loss = Lambda(model_loss_lambda, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
            
        model = Model([model_body.input, *y_true], model_loss)
    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(model_loss_lambda, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_locloss_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', grid_loss=False):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    # batch_details = K.placeholder(shape=(1,))

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    if grid_loss: 
        model_loss_total = Lambda(model_grid_loss_lambda, output_shape=(1, ), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_xy_0 = Lambda(model_grid_loss_xy_0_lambda, output_shape=(3, ), name='yolo_loss_xy_0',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_xy_1 = Lambda(model_grid_loss_xy_1_lambda, output_shape=(3, ), name='yolo_loss_xy_1',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_xy_2 = Lambda(model_grid_loss_xy_2_lambda, output_shape=(3, ), name='yolo_loss_xy_2',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
            
        model_loss_wh_0 = Lambda(model_grid_loss_wh_0_lambda, output_shape=(3, ), name='yolo_loss_wh_0',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_wh_1 = Lambda(model_grid_loss_wh_1_lambda, output_shape=(3, ), name='yolo_loss_wh_1',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_wh_2 = Lambda(model_grid_loss_wh_2_lambda, output_shape=(3, ), name='yolo_loss_wh_2',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_class_0 = Lambda(model_grid_loss_class_0_lambda, output_shape=(3, ), name='yolo_loss_class_0',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_class_1 = Lambda(model_grid_loss_class_1_lambda, output_shape=(3, ), name='yolo_loss_class_1',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_class_2 = Lambda(model_grid_loss_class_2_lambda, output_shape=(3, ), name='yolo_loss_class_2',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_confidence_0 = Lambda(model_grid_loss_confidence_0_lambda, output_shape=(3, ), name='yolo_loss_confidence_0',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_confidence_1 = Lambda(model_grid_loss_confidence_1_lambda, output_shape=(3, ), name='yolo_loss_confidence_1',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_loss_confidence_2 = Lambda(model_grid_loss_confidence_2_lambda, output_shape=(3, ), name='yolo_loss_confidence_2',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_output_0 = Lambda(model_output_0_lambda, output_shape=(3, ), name='yolo_output_0',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_output_1 = Lambda(model_output_1_lambda, output_shape=(3, ), name='yolo_output_1',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_output_2 = Lambda(model_output_2_lambda, output_shape=(3, ), name='yolo_output_2',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model_object_mask_0 = Lambda(model_object_mask_0_lambda, output_shape=(3, ), name='yolo_obj_mask_0',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
        
        model_object_mask_1 = Lambda(model_object_mask_1_lambda, output_shape=(3, ), name='yolo_obj_mask_1',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
            
        model_object_mask_2 = Lambda(model_object_mask_2_lambda, output_shape=(3, ), name='yolo_obj_mask_2',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])

        model = Model([model_body.input, *y_true], [model_loss_xy_0, model_loss_wh_0, model_loss_class_0, model_loss_confidence_0, 
        model_loss_xy_1, model_loss_wh_1, model_loss_class_1, model_loss_confidence_1, 
        model_loss_xy_2, model_loss_wh_2, model_loss_class_2, model_loss_confidence_2,
        model_loss_total, model_output_0, model_output_1, model_output_2, model_object_mask_0, model_object_mask_1, model_object_mask_2])
    else:
        model_loss = Lambda(model_loss_lambda, output_shape=(1,), name='yolo_loss',
            arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
            [*model_body.output, *y_true])
            
        model = Model([model_body.input, *y_true], model_loss)
    return model

    
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        batch_data = []
        for b in range(batch_size):
            if i==0:
                pass
                # np.random.shuffle(annotation_lines) removing random shuffle for now
            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            batch_data.append(annotation_lines[i])
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes, batch_data)
        print('len(y_true): ', len(y_true))
        # print('Batch data:', y_true[-1])
        yield [image_data, *y_true], np.zeros(batch_size)
        # print('Generated batch:', batch_data)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
