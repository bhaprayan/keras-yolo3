"""
Code to process Nuro QA reports and generate a CSV file /w

UUID | Frame Number | x min | y min | x max | y max
"""

import glob
import json

import ipdb
import pandas as pd


def compute_max_form(height,width,left,top):
    br_y, br_x, tl_y, tl_x = top+height, left+width, top, left
    return tl_x, tl_y, br_x, br_y

def ret_mapping(qa_report='qa_report.json'):
    f = json.load(open(qa_report))
    frames = []
    for i in range(len(f['nuro_labels'])): 
        cam = f['nuro_labels'][i] 
        for j in range(len(cam['frames'])): 
            frames.extend(cam['frames'])
    scaleid_list = []
    frameid_list = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for fn in range(len(frames)):
        for annotation in range(len(frames[fn])):
            try:
                frame_idx = frames[fn]['linking_frame_idx']
                annot = frames[fn]['annotations'][annotation]
                scaleid_list.append(annot['scale_id'])
                frameid_list.append(frame_idx)
                tl_x, tl_y, br_x, br_y = compute_max_form(annot['bbox']['height'], annot['bbox']['width'], annot['bbox']['left'], annot['bbox']['top'])
                xmin.append(tl_x)
                ymin.append(tl_y)
                xmax.append(br_x)
                ymax.append(br_y)
            except Exception as e:
                print(e)
    return (scaleid_list, frameid_list, xmin, ymin, xmax, ymax)

if __name__ == "__main__":
    scaleid_all, frameid_all, xmin_all, ymin_all, xmax_all, ymax_all = [], [], [], [], [], []
    qa_reports = glob.glob('../quac/reports/*')
    for report in qa_reports:
        print('Processing:', report)    
        scaleid_list, frameid_list, xmin, ymin, xmax, ymax = ret_mapping(report)
        scaleid_all.extend(scaleid_list)
        frameid_all.extend(frameid_list)
        xmin_all.extend(xmin)
        ymin_all.extend(ymin)
        xmax_all.extend(xmax)
        ymax_all.extend(ymax)
    dat = pd.DataFrame()
    dat['uuid'], dat['frame_no'], dat['xmin_gt'], dat['ymin_gt'], dat['xmax_gt'], dat['ymax_gt'] = scaleid_all, frameid_all, xmin_all, ymin_all, xmax_all, ymax_all
    dat = dat.set_index('uuid')
    dat.to_csv('nuro_gt.txt', sep='\t')
