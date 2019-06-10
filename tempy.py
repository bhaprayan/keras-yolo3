from train import *
from keras.models import load_model
from PIL import Image
from yolo3.utils import letterbox_image
from yolo import YOLO, detect_video
import matplotlib.pyplot as plt
import glob

annotation_path = 'train_nuro.txt'
log_dir = 'logs/000/'
classes_path = 'model_data/classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
num_anchors = len(anchors)

input_shape = (416,416) # multiple of 32, hw

model = create_model(input_shape, anchors, num_classes, 
freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

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
model.compile(optimizer=Adam(lr=1e-3), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})

batch_size = 32
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

x = data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)

data = x.__next__()

images = data[0]
# labels = data[1] these are not labels. batch size

model_path = 'logs/000/ep009-loss30.814-val_loss30.951.h5'

yolo_model =  yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
yolo_model.load_weights(model_path)

FLAGS = {
        "model_path": model_path,
        "anchors_path": anchors_path,
        "classes_path": classes_path,
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

model = YOLO(**vars(FLAGS))

image_base_path = '/home/shuby.deshpande/workspace/keras-yolo3/data/5cc3a5ef4e436f43f7b5615f/images'
image_paths = glob.glob(image_base_path)
for image_path in image_paths:
    image = Image.open(image_path)
    pred = model.detect_image(image)
    image_name = image_path.split('.')[0] + '_pred.jpeg'
    print('-'*50)
    print('Predicting bbox for image:', image_path)
    print('-'*50)
    plt.imsave(image_name, pred)
