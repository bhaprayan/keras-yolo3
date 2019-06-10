from train import *
annotation_path = 'train_nuro.txt'
log_dir = 'logs/000/'
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
        freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
model.compile(optimizer=Adam(lr=1e-3), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})

batch_size = 32
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
x = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
x.__attr__
dir(x)
x()
x = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
x
type(x)
x.__next__()
dir(x)
data = x.__next__()
data.shape
len(data)
data[0].shape
images = data[0]
images[0]
images[0].shape
images
len(images)
len(images)
data
labels = data[1]
len(labels)
labels.shape
class_names
images.shape
images
images[0].shape
images[1].shape
images[2].shape
images[3].shape
len(images)
images[0].shape
images[1].shape
images[1][...,:]
images[1][...,:].shape
images[1].shape
images[1][0,0,0,0,:].shape
images[1][0,0,0,0,:]
data
len(data)
data
model.fit(data, steps_per_epoch=max(1, num_train//batch_size), epochs=50, initial_epoch=0)
data[0]
images
images.shape
images[1].shape
images[0].shape
images[...,0].shape
images[0][...,0].shape
images[1][...,:].shape
images[1][0,0,0,0,:].shape
images[1][0,0,0,0,:]
images[0].shape
images[1].shape
images[1].shape
from keras.models import load_model
self.yolo_model = load_model(model_path, compile=False)
ls
ls logs/000/
model_path = 'logs/000/ep009-loss30.814-val_loss30.951.h5'
self.yolo_model = load_model(model_path, compile=False)
model_path = 'logs/000/ep006-loss30.853-val_loss30.983.h5'
self.yolo_model = load_model(model_path, compile=False)
yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
num_anchors = len(anchors)
yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
self.yolo_mdoel =  yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
self.yolo_model =  yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
yolo_model =  yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
yolo_model.load_weights(model_path)
yolo_models
yolo_model
yolo_model.input
K.learning_phase?
yolo_model.losses
from PIL import Image
image_path = '/home/shuby.deshpande/workspace/keras-yolo3/data/5cc3a5ef4e436f43f7b5615f/images/0.jpeg'
image = Image.open(image_path)
image.shape
image.size
model_image_size = (416, 416)
if model_image_size != (None, None):
    assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
    assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
    boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
else:
    new_image_size = (image.width - (image.width % 32),
                        image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
from yolo3.utils import letterbox_image
if model_image_size != (None, None):
    assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
    assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
    boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
else:
    new_image_size = (image.width - (image.width % 32),
                        image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
image_data.shaep
image_data.shape
image_data
boxed_image.shape
boxed_image.size
print(image_data.shape)
image_data = np.array(boxed_image, dtype='float32')
print(image_data.shape)
image_data /= 255.
image_data = np.expand_dims(image_data, 0)
image_data.shape
out_boxes, out_scores, out_classes = self.sess.run(
    [self.boxes, self.scores, self.classes],
    feed_dict={
        self.yolo_model.input: image_data,
        self.input_image_shape: [image.size[1], image.size[0]],
        K.learning_phase(): 0
    })
ls
%history tempy.py
ls
%history -f tempy.py
