import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from .combined_thresh import combined_thresh
from .perspective_transform import perspective_transform
from .Line import Line
from .line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip
import sys
import argparse
import os


# Global variables (just to make the moviepy video annotation work)
file_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(file_dir, 'calibrate_camera.p'), 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 5  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes
left_lane_inds, right_lane_inds = None, None  # for calculating curvature


# MoviePy video annotation will call this function
def annotate_image(img_in, visual_target=None):
    """
    Annotate the input image with lane line markings
    Returns annotated image
    """
    global mtx, dist, left_line, right_line, detected
    global left_curve, right_curve, left_lane_inds, right_lane_inds

    # Undistort, threshold, perspective transform
    undist = cv2.undistort(img_in, mtx, dist, None, mtx)
    un_distorted_target = cv2.undistort(visual_target, mtx, dist, None, mtx)

    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
    binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)

    # Perform polynomial fit
    if not detected:
        # Slow line fit
        ret = line_fit(binary_warped)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        nonzerox = ret['nonzerox']
        nonzeroy = ret['nonzeroy']
        left_lane_inds = ret['left_lane_inds']
        right_lane_inds = ret['right_lane_inds']

        # Get moving average of line fit coefficients
        left_fit = left_line.add_fit(left_fit)
        right_fit = right_line.add_fit(right_fit)

        # Calculate curvature
        left_curve, right_curve = calc_curve(
            left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

        detected = True  # slow line fit always detects the line

    else:  # implies detected == True
        # Fast line fit
        left_fit = left_line.get_fit()
        right_fit = right_line.get_fit()
        ret = tune_fit(binary_warped, left_fit, right_fit)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        nonzerox = ret['nonzerox']
        nonzeroy = ret['nonzeroy']
        left_lane_inds = ret['left_lane_inds']
        right_lane_inds = ret['right_lane_inds']

        # Only make updates if we detected lines in current frame
        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            left_curve, right_curve = calc_curve(
                left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
        else:
            detected = False

    vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

    # Perform final visualization on top of original undistorted image
    result_ = final_viz(un_distorted_target, left_fit, right_fit, m_inv,
                        left_curve, right_curve, vehicle_offset)

    return result_


def annotate_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(annotate_image)
    annotated_video.write_videofile(output_file, audio=False)


def parse_args():
    arg_parser = argparse.ArgumentParser('demo of lane detect')
    arg_parser.add_argument('--images', default='./test_images/', help='images dir.')
    arg_parser.add_argument('--video', default=None, help='video file.')
    return arg_parser.parse_args()

if __name__ == '__main__':
    # Annotate the video
    args = parse_args()

    images_dir = args.images
    video_f = args.video

    if video_f:
        # do video predict
        cap = cv2.VideoCapture(video_f)
        print('# predict on video..')

        i = 0
        while cap.isOpened():
            # Load images from video and crop
            ret, frame = cap.read()
            if ret:
                i += 1
                result = annotate_image(frame)
                video_f_name = os.path.basename(video_f).split('.')[0]
                save_dir = './results/video/{}'.format(video_f_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, 'frame_%04d.jpg' % i), result)
                cv2.imshow('video', result)
                cv2.waitKey(1)
    else:
        # do images
        images = [os.path.join(images_dir, i) for i in os.listdir(images_dir) if i.endswith('jpg')]
        for img in images:
            img = cv2.imread(img, cv2.CAP_MODE_RGB)
            img = annotate_image(img)
            cv2.imshow('image', img)
            cv2.waitKey(0)

