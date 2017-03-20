#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:04:55 2017

@author: HimanshuPal
"""

# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from sklearn import datasets, linear_model
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    # return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lane_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    imshape = image.shape

    gray = grayscale(image)

    blurred_mage = gaussian_blur(gray, kernel_size=7)

    img_with_canny_edges = canny(blurred_mage, 50, 150)

    vertices = np.array([[(0, imshape[0]), (465, 320), (475, 320), (imshape[1], imshape[0])]], dtype=np.int32)

    polygon = region_of_interest(img_with_canny_edges, vertices)

    # Hough lines
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 40  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments
    lines = hough_lines(polygon, rho, theta, threshold, min_line_len, max_line_gap)

    vertices = np.array([[(0, imshape[0]), (200, 320), (800, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    lines = region_of_interest(lines, vertices)

    result = weighted_img(lines, image)

    return result


def get_slope(line):
    """ Calculates the slope of a line """
    for x1, y1, x2, y2 in line:
        return (y1 - y2) / (x1 - x2)


def get_line_size(line):
    """ Calculates and returns the size in pixels of a line"""
    for x1, y1, x2, y2 in line:
        return math.hypot(x2 - x1, y2 - y1)


def divide_lines(lines):
    """ Divides the relevant hough lines into left & right sections and ignores the noise"""
    right_lines = []
    left_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = get_slope(line)
            if slope > 0.5:
                right_lines.append([x1, y1, x2, y2, slope])
            if slope < -0.5:
                left_lines.append([x1, y1, x2, y2, slope])

    return right_lines, left_lines

def moving_average(previous_avg, new_position, N=20):
    """ Calculates the moving average of a co-ordinate of a line"""
    if (previous_avg == 0):
        return new_position
    previous_avg -= previous_avg / N;
    previous_avg += new_position / N;
    return previous_avg;


def extend_line(x1, y1, x2, y2, length):
    """ Takes line endpoints and extroplates new endpoint by a specfic length"""
    line_len = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    x = x2 + (x2 - x1) / line_len * length
    y = y2 + (y2 - y1) / line_len * length
    return int(x), int(y)


def draw_right_lane_line(img, right_lines):
    global avg_top_right_x
    global avg_top_right_y
    global avg_bottom_right_x
    global avg_bottom_right_y

    x1 = moving_average(avg_top_right_x, int(np.mean(right_lines, axis=0)[0]))
    y1 = moving_average(avg_top_right_y, int(np.mean(right_lines, axis=0)[1]))
    x2 = moving_average(avg_bottom_right_x, int(np.mean(right_lines, axis=0)[2]))
    y2 = moving_average(avg_bottom_right_y, int(np.mean(right_lines, axis=0)[3]))

    avg_top_right_x = x1
    avg_top_right_y = y1
    avg_bottom_right_x = x2
    avg_bottom_right_y = y2

    x1, y1 = extend_line(x1, y1, x2, y2, -1000)
    x2, y2 = extend_line(x1, y1, x2, y2, 1000)

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)


def draw_left_lane_line(img, left_lines):
    global avg_top_left_x
    global avg_top_left_y
    global avg_bottom_left_x
    global avg_bottom_left_y

    x1 = moving_average(avg_top_left_x, int(np.mean(left_lines, axis=0)[0]))
    y1 = moving_average(avg_top_left_y, int(np.mean(left_lines, axis=0)[1]))
    x2 = moving_average(avg_bottom_left_x, int(np.mean(left_lines, axis=0)[2]))
    y2 = moving_average(avg_bottom_left_y, int(np.mean(left_lines, axis=0)[3]))

    avg_top_left_x = x1
    avg_top_left_y = y1
    avg_bottom_left_x = x2
    avg_bottom_left_y = y2

    x1, y1 = extend_line(x1, y1, x2, y2, -1000)
    x2, y2 = extend_line(x1, y1, x2, y2, 1000)

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)


def draw_lane_lines(img, lines):
    right_lines, left_lines = divide_lines(lines)
    draw_left_lane_line(img, left_lines)
    draw_right_lane_line(img, right_lines)


avg_top_right_x = 0
avg_top_right_y = 0
avg_bottom_right_x = 0
avg_bottom_right_y = 0

avg_top_left_x = 0
avg_top_left_y = 0
avg_bottom_left_x = 0
avg_bottom_left_y = 0

directory = '/Users/sitaraassomull/Code/udacity/nanodegree/p1-project/'
img_folder = directory + 'test_images/'

imshape = []
imgs = []
for img in os.listdir(img_folder):
    test_image = mpimg.imread(img_folder + img)
    result = process_image(test_image)
    plt.imshow(result)
    plt.show()

white_output = directory + 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip(directory + "test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

white_output = directory + 'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip(directory + "test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
