import numpy as np
import os
from scipy.ndimage import interpolation as inter
import cv2
import imutils


def find_score(arr, angle):
    data = inter.rotate(arr, angle,reshape=False, order=0)
    hist = np.sum(data, axis=1)
    scores = (hist[1:] - hist[:-1]) ** 3
    scores = [abs(s) for s in scores]
    score = np.sum(scores)
    score = abs(score)
    return hist, score

def get_rel_snippet(bin_img):
    [h, w] = bin_img.shape[:2]
    h_10 = int(0.10 * h)
    h_30 = int(0.30 * h)
    h_35 = int(0.35 * h)
    w_10 = int(0.15 * w)
    h_5 = int(0.05*h)
    if sum(sum(bin_img[h_35:(h-h_30),w_10:(w-w_10)]))>50000:
        return bin_img[h_35:(h-h_30),w_10:(w-w_10)]
    elif sum(sum(bin_img[h_10:(h-h_35),w_10:(w-w_10)]))>50000:
        return bin_img[h_10:(h-h_35),w_10:(w-w_10)]
    elif sum(sum(bin_img[h_5:(h-h_35),w_10:(w-w_10)]))>50000:
        return bin_img[h_10:(h-h_35),w_10:(w-w_10)]
    return bin_img[h_35:(h-h_10),w_10:(w-w_10)]

def orient(img):
    bin_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    [h,w] = bin_img.shape[:2]
    bin_img = get_rel_snippet(bin_img)
    bin_img = cv2.dilate(bin_img,np.ones((2,2)))
    bin_img = bin_img/255
    delta = 5
    limit = 40
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        # print("angle",angle,"score",score)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle - first run: ',best_angle)

    delta = 1
    angles = np.arange(best_angle-5, best_angle+5, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        # print("angle",angle,"score",score)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    deskew_im = imutils.rotate_bound(img,-1*best_angle)
    return deskew_im

img = cv2.imread('sample2.jpg',0)
res = orient(img)
cv2.imwrite('sample2_res.jpg',res)



