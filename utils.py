import logging
import logging.config
import argparse

import yaml


def get_logger(logger_name):
    with open('logger.yml', 'r') as f:
        log_cfg = yaml.safe_load(f.read())

    logging.config.dictConfig(log_cfg)
    return logging.getLogger(logger_name)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-image-folder", type=str, default='../data/train')
    parser.add_argument("--train-csv-path", type=str, default='../data/train.csv')
    parser.add_argument("--test-image-folder", type=str, default='../data/test')
    parser.add_argument("--test-csv-path", type=str, default='../data/test.csv')
    parser.add_argument("--model-path", type=str, default='../model_new.pt')
    parser.add_argument("--epochs", type=int, default=2)
    return parser


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
