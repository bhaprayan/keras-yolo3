import json

f = json.load(open('qa_report.json'))
frames = []

for i in range(len(f['nuro_labels'])): 
    cam = f['nuro_labels'][i] 
    for j in range(len(cam['frames'])): 
        frames.extend(cam['frames'])

mapping = {}

for fn in range(len(frames)):
    for annotation in range(len(frames[fn])):
        try:
            mapping[frames[fn]['annotations'][annotation]['scale_id']] = frames[fn]['annotations'][annotation]['bbox'] 
        except:
            continue
