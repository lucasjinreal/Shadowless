"""
detect manger manage various detect methods.
be careful that, if we using MXNet then avoid using TensorFlow!!!
they can not be used in same time!!!

"""


class DetectManager(object):

    def __init__(self, core='ssd'):
        self.core = core

        if self.core == 'ssd':
            # from .mxnet_ssd.mxnet_ssd_detector import MXNetSSDDetector
            from .mxnet_ssd.demo import SSDDetector
            print('[DETECT] ssd backend.')
            self.detector = SSDDetector()
        elif self.core == 'mask-rcnn':
            from .mask_rcnn.mask_rcnn_detector import MaskRCNNDetector
            print('[DETECT] mask-rcnn backend.')
            self.detector = MaskRCNNDetector()

    def run_detect(self, img, visual_target):
        # run a single detect on img
        return self.detector.run_detect(img, visual_target=visual_target)



