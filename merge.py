import pandas as pd
from process_utils import iou, writer, return_annotate_dict

labeled_dat = pd.read_csv('labeled_nuro.txt', sep='\t')
gt_dat = pd.read_csv('nuro_gt.txt', sep='\t')

annotation_path = 'updated_train_nuro.txt'
uuid_path = 'updated_uuid_nuro.txt'

merged_dat = gt_dat.merge(labeled_dat)

merged_dat['iou'] = merged_dat.apply(lambda row: iou((row['xmin_gt'], row['ymin_gt'], row['xmax_gt'], row['ymax_gt']), 
    (row['x_min'], row['y_min'], row['x_max'], row['y_max'])), axis=1)

print('Computed IOU!')
# merged_dat.to_csv('merged_dat.txt', sep='\t')

# merged_dat = pd.read_csv('merged_dat.txt', sep='\t')

# iou_gt_paths = list(set(merged_dat[merged_dat['iou'] > 0.85]['image_path']))

# iou_lt_paths = list(set(merged_dat[merged_dat['iou'] <= 0.85]['image_path']))

all_paths = list(set(merged_dat['image_path']))

# writer('iou_gt_paths', iou_gt_paths)
# writer('iou_lt_paths', iou_lt_paths)

annotate_list = open(annotation_path).readlines()
uuid_list = open(uuid_path).readlines() 

annotate_dict = return_annotate_dict(annotation_path)
print('Returned dict')
# good_label_train = open('good_label_train_nuro.txt', 'w')
# good_label_uuid = open('good_label_uuid_nuro.txt', 'w') 
# bad_label_train = open('bad_label_train_nuro.txt', 'w')
# bad_label_uuid = open('bad_label_uuid_nuro.txt', 'w')

total_label_train = open('total_label_train_nuro.txt', 'w') 
total_label_uuid = open('total_label_uuid_nuro.txt', 'w') 

# for path in iou_gt_paths:
#     idx = annotate_dict[path]
#     annotation = annotate_list[idx]
#     uuid = uuid_list[idx]
#     good_label_train.write(annotation)
#     good_label_uuid.write(uuid)

# for path in iou_lt_paths:
#     idx = annotate_dict[path]
#     annotation = annotate_list[idx]
#     uuid = uuid_list[idx]
#     bad_label_train.write(annotation)
#     bad_label_uuid.write(uuid)

for path in all_paths:
    idx = annotate_dict[path]
    annotation = annotate_list[idx]
    uuid = uuid_list[idx]
    total_label_train.write(annotation)
    total_label_uuid.write(uuid)

# good_label_train.close()
# good_label_uuid.close()
# bad_label_train.close()
# bad_label_uuid.close()
