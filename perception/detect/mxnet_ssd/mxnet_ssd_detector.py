"""
this is the wrapper of mxnet ssd detector
"""
import argparse
from .tools import find_mxnet
import mxnet as mx
import os
import sys
from .detect.detector import Detector
from .symbol.symbol_factory import get_symbol


class MXNetSSDDetector(object):
    def __init__(self):
        self.net = 'resnet50'
        self.data_shape = 512
        self.mean_pixels = [123, 117, 104]
        self.ctx = mx.gpu(0)

        self.classes = ['aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor']
        self.num_classes = len(self.classes)
        self.nms_thresh = 0.5
        self.thresh = 0.5
        self.force_nms = True
        self.nms_topk = 400

        self.prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'model', 'ssd_{}_{}'.format(self.net, self.data_shape))
        self.epoch = 0

        self.detector = self._get_detector()

    def _get_detector(self):
        args_ = self.parse_args()
        if args_.cpu:
            ctx_ = mx.cpu()
        else:
            ctx_ = mx.gpu(args_.gpu_id)

        network_ = None if args_.deploy_net else args_.network
        self.classes = self.parse_class_names(args_.class_names)
        if args_.prefix.endswith('_'):
            prefix_ = args_.prefix + args_.network + '_' + str(args_.data_shape)
        else:
            prefix_ = args_.prefix
        detector_ = self.get_detector(network_, prefix_, args_.epoch,
                                      args_.data_shape,
                                      (args_.mean_r, args_.mean_g, args_.mean_b),
                                      ctx_, len(self.classes), args_.nms_thresh, args_.force_nms)
        return detector_

    def run_detect(self, img, visual_target):
        result_img, detection = self.detector.detect_on_single_image(img=img, classes=self.classes,
                                                                     thresh=self.thresh, visual_target=visual_target)
        return result_img, detection

        # return self.detector.detect_on_single_image(img, self.classes, self.nms_thresh, visual_target)

    @staticmethod
    def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                     nms_thresh=0.5, force_nms=True, nms_topk=400):
        if net is not None:
            net = get_symbol(net, data_shape, num_classes=num_class, nms_thresh=nms_thresh,
                             force_nms=force_nms, nms_topk=nms_topk)
        detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
        return detector

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Single-shot detection network demo')
        parser.add_argument('--network', dest='network', type=str, default='resnet50',
                            help='which network to use')
        parser.add_argument('--images', dest='images', type=str,
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/demo'),
                            help='run demo with images, use comma to seperate multiple images')
        parser.add_argument('--video', type=bool, default=False, help='Bool to set video or not.')
        parser.add_argument('--dir', dest='dir', nargs='?',
                            help='demo image directory, optional', type=str)
        parser.add_argument('--ext', dest='extension', help='image extension, optional',
                            type=str, nargs='?')
        parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                            default=0, type=int)
        parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'ssd_'),
                            type=str)
        parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                            action='store_true', default=False)
        parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                            help='GPU device id to detect with')
        parser.add_argument('--data-shape', dest='data_shape', type=int, default=512,
                            help='set image shape')
        parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                            help='red mean value')
        parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                            help='green mean value')
        parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                            help='blue mean value')
        parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                            help='object visualize score threshold, default 0.6')
        parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                            help='non-maximum suppression threshold, default 0.5')
        parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                            help='force non-maximum suppression on different class')
        parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                            help='show detection time')
        parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                            help='Load network from json file, rather than from symbol')
        parser.add_argument('--class-names', dest='class_names', type=str,
                            default='aeroplane, bicycle, bird, boat, bottle, bus, \
                            car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                            person, pottedplant, sheep, sofa, train, tvmonitor',
                            help='string of comma separated names, or text filename')
        args = parser.parse_args()
        return args

    @staticmethod
    def parse_class_names(class_names):
        """ parse # classes and class_names if applicable """
        if len(class_names) > 0:
            if os.path.isfile(class_names):
                # try to open it to read class names
                with open(class_names, 'r') as f:
                    class_names = [l.strip() for l in f.readlines()]
            else:
                class_names = [c.strip() for c in class_names.split(',')]
            for name in class_names:
                assert len(name) > 0
        else:
            raise RuntimeError("No valid class_name provided...")
        return class_names
