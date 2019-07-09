def iou(labeled, gt):
    """
    labeled: (x_min y_min x_max y_max)
    gt: (xmin_gt ymin_gt xmax_gt ymax_gt)
    """
    x_min, y_min, x_max, y_max = labeled
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt
    x1, y1 = max(x_min, xmin_gt), max(y_min, ymin_gt)
    x2, y2 = min(x_max, xmax_gt), max(y_max, ymax_gt)
    width, height = x2-x1, y2-y1
    if(width > 0 and height > 0):
        overlap = width * height
    else:
        overlap = 0
    a1 = (x_max - x_min) * (y_max - y_min)
    a2 = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
    combined_region = a1 + a2 - overlap
    iou = overlap/combined_region
    return iou

def writer(fn, data):
    # TODO: generalize code to save lists, dicts, etc.
    with open(fn, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

def return_annotate_dict(annotation_path):
    """
    Use the annotate dict to return the line number from the train file. Same for uuid file 
    """
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    annotate_dict = {}
    for i, line in enumerate(annotation_lines):
        annotate_dict[line.split()[0]] = i
    return annotate_dict
