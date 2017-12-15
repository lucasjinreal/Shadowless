"""
Mono Camera manager,
this file will open a camera and returns the results of
detections, lane information, and segmentation results

using this in:

mono_manager = MonoManager()
mono_manager.serve()
this will open a while loop to read from camera,

this also can load a video in local by send:
mono_manager.serve_local(video_file)

we will call
"""
import cv2
import os
from ..detect.detect_manager import DetectManager
from ..lane.lane_manager import LaneManager


class MonoManager(object):

    def __init__(self):
        self.detect_manager = DetectManager(core='ssd')
        self.lane_manager = LaneManager()

    def do_detect(self, img, visual_target):
        return self.detect_manager.run_detect(img, visual_target)

    def do_segment(self, img):
        pass

    def do_lane_detect(self, img, visual_target):
        return self.lane_manager.run_lane_detect(img, visual_target)

    def serve(self):
        pass

    def serve_local(self, video_f, is_record=False):
        if not os.path.exists(video_f):
            print('# can not find this video file.')
        cap = cv2.VideoCapture(video_f)
        print('# predict on video..')

        i = 0
        while cap.isOpened():
            # Load images from video and crop
            ret, frame = cap.read()
            if ret:
                i += 1

                result, _ = self.do_detect(img=frame, visual_target=frame)
                result = self.do_lane_detect(img=frame, visual_target=result)

                if is_record:
                    video_f_name = os.path.basename(video_f).split('.')[0]
                    save_dir = './results/video_record/{}'.format(video_f_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cv2.imwrite(os.path.join(save_dir, 'frame_%04d.jpg' % i), result)
                cv2.imshow('video', result)
                cv2.waitKey(1)

