import logging
import logging.config
import copy
from PIL import ImageDraw, ImageFont

import yaml


def get_logger(logger_name):
    with open('logger.yml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())

    logging.config.dictConfig(log_cfg)
    return logging.getLogger(logger_name)


def get_iou(bb1, bb2):
    xmin1, xmax1, ymin1, ymax1 = bb1
    xmin2, xmax2, ymin2, ymax2 = bb2

    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    bb2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    iou = abs(intersection_area) / float(abs(bb1_area) + abs(bb2_area) - abs(intersection_area))
    return iou


def get_image_with_bbox_and_text(img, bbox, text):
    img = copy.deepcopy(img)
    img = img.resize((184 * 2, 112 * 2))

    xmin, xmax, ymin, ymax = bbox

    x0 = (xmin) * img.width
    y0 = (ymin) * img.height

    x1 = (xmax) * img.width
    y1 = (ymax) * img.height

    draw = ImageDraw.Draw(img)
    bbox_abs = [(x0, y0), (x1, y1)]

    draw.rectangle(bbox_abs, fill=None, outline='red', width=2)
    font = ImageFont.truetype('Helvetica', size=16)
    draw.text((x0, y0 - 16), text, fill='blue', font=font)
    return img
