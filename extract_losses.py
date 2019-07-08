import json
import glob

fs = glob.glob('*yolo_data.json')
uuid_scales = ['0_uuid', '1_uuid', '2_uuid']
xy_loss_scales = ['0_xy_model_loss', '1_xy_model_loss', '2_xy_model_loss']
wh_loss_scales = ['0_wh_model_loss', '1_wh_model_loss', '2_wh_model_loss']
for fn in fs[:1]:
    data = json.load(open(fn))
    uuids = []
    x_loss = []
    y_loss = []
    w_loss = []
    h_loss = []
    for uuid_scale_idx in range(len(uuid_scales)):
        if(data[uuid_scales[uuid_scale_idx]]):
            uuids.append(*data[uuid_scales[uuid_scale_idx]])
            x_loss.append(*data[xy_loss_scales[uuid_scale_idx]][0::2]) #x loss is 1st idx
            y_loss.append(*data[xy_loss_scales[uuid_scale_idx]][1::2]) #y loss is 2nd idx
            w_loss.append(*data[wh_loss_scales[uuid_scale_idx]][0::2]) #w loss is 1st idx
            h_loss.append(*data[wh_loss_scales[uuid_scale_idx]][1::2]) #h loss is 2nd idx