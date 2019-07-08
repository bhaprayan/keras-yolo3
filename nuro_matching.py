import json
import shlex, subprocess 
import re

train_file = open('updated_train_nuro.txt').readlines()
uuid_file = open('updated_uuid_nuro.txt').readlines()

def grepper(fn, subtask_idx):
    lines = []
    for l in fn:
        if re.search(subtask_idx,l):
            lines.append(l) 
    return lines

def write_filtered(data, fn):
    with open(fn, 'w') as f:
        for line in data:
            f.write("%s" % line)
        f.close()

qa_reports = open('qa_reports.txt').read().split()

subtasks = []

for report in qa_reports:
    # adding try except since some reports differ in structure i.e.
    # TODO: why the difference?
    # NOTE: reading out subtasks from here since the qa_reports file doesn't have the subtask ID
    # https://s3-us-west-1.amazonaws.com/6876.qa.report.linking/20190115_160816_00023_2646.0_2676.0_linking/qa_report.json
    # https://scale.ai/corp/compare_attempts?subtask=5cfe774faaf51a0e1c8570a7&customerTaskResponse=https://s3-us-west-1.amazonaws.com/6876.qa.report.linking/20190115_160816_00023_2646.0_2676.0_linking/qa_report.json
    try:
        subtask_idx = report.split('?')[1].split('&')[0].split('=')[1]
        subtasks.append(subtask_idx)
    except:
        continue

for subtask_idx in subtasks[:1]:
    SUBTASK_ID = subtask_idx
    data = grepper(train_file, SUBTASK_ID)
    fn_train = 'filtered_subtask/subtask_'+SUBTASK_ID+'_train.txt'
    write_filtered(data, fn_train)
    fn_uuid = 'filtered_subtask/subtask_'+SUBTASK_ID+'_uuid.txt'
    data = grepper(uuid_file, SUBTASK_ID)
    write_filtered(data, fn_uuid)
# extract object id from qa reports file
# grep uuid file for subtask id containing the object
# extract train + uuid lines from this subtask i:
# run model on the filtered subtask
# extract frames corresponding to matching object id


# grep SUBTASK_ID updated_train_nuro.txt > filtered_subtask/subtask_SUBTASK_ID_train.txt
# grep SUBTASK_ID updated_uuid_nuro.txt > filtered_subtask/subtask_SUBTASK_ID_uuid.txt

# f['nuro_labels'][0]['frames'][0]['annotations'][0]['scale_id'] 
