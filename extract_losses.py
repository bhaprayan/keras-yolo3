import json
import glob
import pandas
import ipdb
import csv

fs = glob.glob('*yolo_data.json')
uuid_scales = ['0_uuid', '1_uuid', '2_uuid']
xy_loss_scales = ['0_xy_model_loss', '1_xy_model_loss', '2_xy_model_loss']
wh_loss_scales = ['0_wh_model_loss', '1_wh_model_loss', '2_wh_model_loss']
csv_file = csv.writer(open('subtask_losses.csv', 'w'))
csv_file.writerow(['uuid','x_loss','y_loss','w_loss','h_loss'])

for fn in fs:
    data = json.load(open(fn))
    uuids = []
    x_loss = []
    y_loss = []
    w_loss = []
    h_loss = []
    for uuid_scale_idx in range(len(uuid_scales)):
        if(data[uuid_scales[uuid_scale_idx]]):
            uuids.extend(data[uuid_scales[uuid_scale_idx]])
            x_loss.extend(data[xy_loss_scales[uuid_scale_idx]][0::2]) #x loss is 1st idx
            y_loss.extend(data[xy_loss_scales[uuid_scale_idx]][1::2]) #y loss is 2nd idx
            w_loss.extend(data[wh_loss_scales[uuid_scale_idx]][0::2]) #w loss is 1st idx
            h_loss.extend(data[wh_loss_scales[uuid_scale_idx]][1::2]) #h loss is 2nd idx
    fn_loss = list(zip(uuids,x_loss,y_loss,w_loss,h_loss))
    for row in fn_loss:
        csv_file.writerow(row)
