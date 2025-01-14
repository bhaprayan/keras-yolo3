import json
import glob
import pandas as pd
import ipdb
import csv

fs_bad = glob.glob('bad_labels/*yolo_data.json')
fs_good = glob.glob('good_labels/*yolo_data.json')
uuid_scales = ['0_uuid','1_uuid','2_uuid']
xy_loss_scales = ['0_xy_model_loss','1_xy_model_loss','2_xy_model_loss']
wh_loss_scales = ['0_wh_model_loss','1_wh_model_loss','2_wh_model_loss']
confidence_scales = ['0_confidence_model_loss','1_confidence_model_loss','2_confidence_model_loss']

csv_file = csv.writer(open('bad_subtask_losses.csv','w'))
csv_file.writerow(['uuid','frame_no','subtask','x_loss','y_loss','w_loss','h_loss', 'confidence'])

for fn in fs_bad:
    data = json.load(open(fn))
    uuids = []
    x_loss = []
    y_loss = []
    w_loss = []
    h_loss = []
    confidence = []
    for uuid_scale_idx in range(len(uuid_scales)):
        if(data[uuid_scales[uuid_scale_idx]]):
            uuids.extend(data[uuid_scales[uuid_scale_idx]])
            #x loss is 1st idx
            x_loss.extend(data[xy_loss_scales[uuid_scale_idx]][0::2])
            #y loss is 2nd idx
            y_loss.extend(data[xy_loss_scales[uuid_scale_idx]][1::2])
            #w loss is 1st idx
            w_loss.extend(data[wh_loss_scales[uuid_scale_idx]][0::2])
            #h loss is 2nd idx
            h_loss.extend(data[wh_loss_scales[uuid_scale_idx]][1::2])
            confidence.extend(data[confidence_scales[uuid_scale_idx]])
    frame_n_list = [data['frame_no']] * len(uuids)
    subtask_list = [data['subtask']] * len(uuids) 
    fn_loss = list(zip(uuids, frame_n_list, subtask_list, x_loss, y_loss, 
    w_loss, h_loss, confidence))
    for row in fn_loss:
        csv_file.writerow(row)

dat = pd.read_csv('bad_subtask_losses.csv')
dat['xy_loss'] = dat['x_loss'] + dat['y_loss']
dat['wh_loss'] = dat['w_loss'] + dat['h_loss']
dat = dat.set_index('uuid')

dat.to_csv('bad_label_losses.csv')

# TODO: cleanup this bit later. refactor into function

csv_file = csv.writer(open('good_subtask_losses.csv','w'))
csv_file.writerow(['uuid','frame_no','subtask','x_loss','y_loss','w_loss','h_loss', 'confidence'])

for fn in fs_good:
    data = json.load(open(fn))
    uuids = []
    x_loss = []
    y_loss = []
    w_loss = []
    h_loss = []
    confidence = []
    for uuid_scale_idx in range(len(uuid_scales)):
        if(data[uuid_scales[uuid_scale_idx]]):
            uuids.extend(data[uuid_scales[uuid_scale_idx]])
            x_loss.extend(data[xy_loss_scales[uuid_scale_idx]][0::2]) #x loss is 1st idx
            y_loss.extend(data[xy_loss_scales[uuid_scale_idx]][1::2]) #y loss is 2nd idx
            w_loss.extend(data[wh_loss_scales[uuid_scale_idx]][0::2]) #w loss is 1st idx
            h_loss.extend(data[wh_loss_scales[uuid_scale_idx]][1::2]) #h loss is 2nd idx
            confidence.extend(data[confidence_scales[uuid_scale_idx]])
    frame_n_list = [data['frame_no']] * len(uuids)
    subtask_list = [data['subtask']] * len(uuids) 
    fn_loss = list(zip(uuids,frame_n_list,subtask_list,x_loss,y_loss,w_loss,h_loss, confidence))
    for row in fn_loss:
        csv_file.writerow(row)

dat = pd.read_csv('good_subtask_losses.csv')
dat['xy_loss'] = dat['x_loss'] + dat['y_loss']
dat['wh_loss'] = dat['w_loss'] + dat['h_loss']
dat = dat.set_index('uuid')

dat.to_csv('good_label_losses.csv')
