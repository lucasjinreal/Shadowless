"""
do some lane manager
this is very important, we currently support some lane detect methods

"""
from .lane_udacity.demo import annotate_image
import os
import cv2


class LaneManager(object):

    def __init__(self, core='udacity'):
        self.core = core

    def run_lane_detect(self, img, visual_target):
        if self.core == 'udacity':
            img = annotate_image(img, visual_target=visual_target)
            return img
