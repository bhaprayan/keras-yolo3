from evaluate_back import *
model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=False)
from train import create_model
model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path=model_path, grid_loss=False)
?model.get_losses_for
from yolo3.model import yolo_loss
model.output
from keras import backend as K
sess = K.get_session()
model_image_size = (416,416)
boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
from yolo3.utils import letterbox_image
boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
image = Image.open(img)
from PIL import Image
img = "data/5cc3a5ef4e436f43f7b5615f/images/0.jpeg"
image = Image.open(img)
boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
image_data = np.array(boxed_image, dtype='float32')
image_data /= 255.
image_data = np.expand_dims(image_data, 0)
input_image_shape = K.placeholder(shape=(2, ))
sess.run([model.output], feed_dict={yolo_model.input: image_data,
input_image_shape: [image.size[1], image.size[0]],
K.learning_phase(): 0
})
model.input
n
image, box = get_random_data(lines[0], input_shape, random=False)
image, box = get_random_data(annotation_lines[0], input_shape, random=False)
len(image_data)
len(y_true)
sess.run([model.output], feed_dict={yolo_model.input: [image_data, *y_true],input_image_shape: [image.size[1], image.size[0]],
K.learning_phase(): 0
})
len(y_true)
*y_true
print(*y_true)
sess.run([model.output], feed_dict={ yolo_model.input: [image_data, *y_true],input_image_shape: [image.size[1], image.size[0]], K.learning_phase(): 0
})
sess.run([model.output], feed_dict={ yolo_model.input: [image_data, y_true[:]],input_image_shape: [image.size[1], image.size[0]], K.learning_phase(): 0})
image_data.shape
batch_size
batch_size = 1
sess.run([model.output], feed_dict={
    yolo_model.input: [[image_data, *y_true], np.zeros(batch_size)],
    input_image_shape: [image.size[1], image.size[0]],
    K.learning_phase(): 0
})
np.zeros(batch_size)
sess.run([model.output], feed_dict={
    model.input: [[image_data, *y_true], np.zeros(batch_size)],
    input_image_shape: [image.size[1], image.size[0]],
    K.learning_phase(): 0
})
model.input
image_data
image_data.shape
model.input
y_true[0].shape
len(y_true)
len([[image_data, y_true[:]]])
len([image_data, y_true[:]])
len([image_data, *y_true[:]])
len([image_data, *y_true])
sess.run([model.output], feed_dict={
    model.input: [image_data, *y_true],
    input_image_shape: [image.size[1], image.size[0]],
    K.learning_phase(): 0
})
model.losses
model.layers
?model.evaluate
sess.run([model.output], feed_dict={
    model.input: [image_data, *y_true],
    input_image_shape: [image.size[1], image.size[0]],
    K.learning_phase(): 0
})
sess.run([model.output], feed_dict={
    model.input: [image_data, *y_true]
})
model.output
model.input
model.evaluate()
model.evaluate(x=[image_data,*y_true])
model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred})
model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
model.weights
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
model.evaluate(x=[image_data,*y_true])
sess.run([model.output], feed_dict={
    model.input: [image_data, *y_true]
})
type(image_data)
type(y_true)
type(*y_true)
model.input
sess.run([model.output], feed_dict={
    i:d for i, d in zip(model.input, [image_data, *y_true])
})
