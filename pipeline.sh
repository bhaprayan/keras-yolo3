#!/bin/bash
source activate scale

#
python generate_dataset.py
# extract relevant fields from the QA reports and dump CSV file
python dump_qa_reports.py

# extract annotations from our JSON files, perform inner join /w the QA reports
# + compute IOU btwn our annotations & QA reports
python merge.py

# argv 1 : location of train_file
# argv 2 : location of uuid_file
python edit_this_file.py good_label_train_nuro.txt good_label_uuid_nuro.txt
mkdir good_labels/
mv *yolo_data.json good_labels/

# argv 1 : location of train_file
# argv 2 : location of uuid_file
python edit_this_file.py bad_label_train_nuro.txt bad_label_uuid_nuro.txt
mkdir bad_labels/
mv *yolo_data.json bad_labels/

# extract the relevant loc losses (xy + wh) from the json files generated in the
# previous step
python extract_losses.py

# train the classifier (loc loss -> label quality)
python train_LR.py
