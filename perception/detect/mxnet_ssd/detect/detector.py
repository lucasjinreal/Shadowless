from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from ..dataset.testdb import TestDB
from ..dataset.iterator import DetIter, VideoIter, ImageIter
import matplotlib.pyplot as plt
import random
import cv2
import colorsys
from collections import namedtuple


class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        print('# loading checkpoints from: ', model_prefix)
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        result = []
        detections = []
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        for pred, _, _ in self.mod.iter_predict(det_iter):
            detections.append(pred[0].asnumpy())
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        for output in detections:
            for i in range(output.shape[0]):
                det = output[i, :, :]
                res = det[np.where(det[:, 0] >= 0)[0]]
                result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2, '{:s} {:.3f}'.format(class_name, score),
                                   bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                   fontsize=12, color='white')
        plt.show()

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """

        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            img = cv2.imread(im_list[k], cv2.CAP_MODE_RGB)
            img = self.visualize_det_cv2(img, det, classes, thresh=thresh)
            cv2.imwrite('./results/result_{}.jpg'.format(k), img)
            cv2.imshow('image', img)
            cv2.waitKey(0)

    def pre_process_data(self, frame_image):
        """
        solve only for inference in video and camera
        :param frame_image:
        :return:
        """
        interp_methods = [cv2.INTER_LINEAR]
        interp_method = interp_methods[int(np.random.uniform(0, 1) * len(interp_methods))]
        frame_image = mx.nd.array(frame_image)
        data = mx.img.imresize(frame_image, self.data_shape, self.data_shape, interp_method)
        data = mx.nd.transpose(data, (2, 0, 1))
        data = data.astype('float32')

        mean_pixels = mx.nd.array(self.mean_pixels).reshape((3, 1, 1))
        data = data - mean_pixels
        return data

    def detect_on_single_image(self, img, classes, thresh, visual_target):
        print('# detect on a image read from cap or video')
        data = self.pre_process_data(img).asnumpy()
        data = np.expand_dims(data, axis=0)
        print('# input data shape: ', data.shape)

        det_iter = mx.io.NDArrayIter(data)
        prediction = self.mod.predict(det_iter)
        detections_one_image = prediction[0].asnumpy()
        print('predictions: ')
        print(detections_one_image)
        print(detections_one_image.shape)
        img = self.visualize_det_cv2(visual_target, detections_one_image,
                                     classes, thresh=thresh, is_video=True)
        return img, detections_one_image

    def detect_on_video(self, video_f, classes, thresh):
        print('# opening video...')
        cap = cv2.VideoCapture(video_f)
        # cap.set(3, 384)
        # cap.set(4, 1248)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        video_iter = VideoIter(video_cap=cap, data_shape=self.data_shape, mean_pixels=self.mean_pixels)

        i = 0
        for pred, _, _ in self.mod.iter_predict(video_iter):
            i += 1
            detections_one_image = pred[0].asnumpy()
            print(detections_one_image)
            print(detections_one_image.shape)
            detections_one_image = np.squeeze(detections_one_image, 0)
            print(detections_one_image.shape)
            current_frame = video_iter.current_frame_image
            print('# current image:', video_iter.current_frame_image.shape)

            img = self.visualize_det_cv2(current_frame, detections_one_image,
                                         classes, thresh=thresh, is_video=True)
            cv2.imshow('image', img)
            cv2.waitKey(1)
            cv2.imwrite('./results/video/frame_%05d.jpg' % i, img)
            print('# image saved.')

    def visualize_det_cv2(self, img, detections, classes=None, thresh=0.6, is_video=False):
        """
        visualize detection on image using cv2, this is the standard way to visualize detections
        :param img:
        :param detections: ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
                each row is one object
        :param classes:
        :param thresh:
        :param is_video
        :return:
        """
        assert classes, 'from visualize_det_cv2, classes must be provided, each class in a list with' \
                        'certain order.'
        assert isinstance(img, np.ndarray), 'from visualize_det_cv2, img must be a numpy array object.'

        height = img.shape[0]
        width = img.shape[1]

        font = cv2.QT_FONT_NORMAL
        font_scale = 0.4
        font_thickness = 1
        line_thickness = 2

        for i in range(detections.shape[0]):
            cls_id = int(detections[i, 0])
            if cls_id >= 0:
                score = detections[i, 1]
                if score > thresh:
                    unique_color = self._create_unique_color_uchar(cls_id)

                    x1 = int(detections[i, 2] * width)
                    y1 = int(detections[i, 3] * height)
                    x2 = int(detections[i, 4] * width)
                    y2 = int(detections[i, 5] * height)

                    cv2.rectangle(img, (x1, y1), (x2, y2), unique_color, line_thickness)

                    text_label = '{} {:.2f}'.format(classes[cls_id], score)
                    (ret_val, base_line) = cv2.getTextSize(text_label, font, font_scale, font_thickness)
                    text_org = (x1, y1 - 0)

                    cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line + 2),
                                  (text_org[0] + ret_val[0] + 5, text_org[1] - ret_val[1] - 2), unique_color,
                                  line_thickness)
                    # this rectangle for fill text rect
                    cv2.rectangle(img, (text_org[0] - 5, text_org[1] + base_line + 2),
                                  (text_org[0] + ret_val[0] + 4, text_org[1] - ret_val[1] - 2),
                                  unique_color, -1)
                    cv2.putText(img, text_label, text_org, font, font_scale, (255, 255, 255), font_thickness)
        return img

    @staticmethod
    def _create_unique_color_float(tag, hue_step=0.41):
        """Create a unique RGB color code for a given track id (tag).

        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.

        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).

        Returns
        -------
        (float, float, float)
            RGB color code in range [0, 1]

        """
        h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
        r, g, b = colorsys.hsv_to_rgb(h, 1., v)
        return r, g, b

    def _create_unique_color_uchar(self, tag, hue_step=0.41):
        """Create a unique RGB color code for a given track id or class in detection (tag).

        The color code is generated in HSV color space by moving along the
        hue angle and gradually changing the saturation.

        Parameters
        ----------
        tag : int
            The unique target identifying tag.
        hue_step : float
            Difference between two neighboring color codes in HSV space (more
            specifically, the distance in hue channel).

        Returns
        -------
        (int, int, int)
            RGB color code in range [0, 255]

        """
        r, g, b = self._create_unique_color_float(tag, hue_step)
        return int(255 * r), int(255 * g), int(255 * b)
