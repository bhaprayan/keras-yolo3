from yolo import YOLO
from PIL import Image
import pandas as pd

dat = open('subset_train_nuro.txt').read().split('\n') 
for line in dat:
    img_name = line.split()[0]
    img = Image.open(img_name)
    model = YOLO()
    predictions_list = model.detect_image(img)
    pred_str = ' '.join((predictions_list))
    print(pred_str)
