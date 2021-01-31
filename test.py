#!/usr/bin/python3

import sys
import cv2
import numpy as np

def preprocess(train_img, query_img):
    image1 = cv2.imread(train_img)
    image2 = cv2.imread(query_img)
    training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    return training_gray, query_gray

def find_match_pts(training_gray, query_gray):
    orb = cv2.ORB_create(10000, 2.0)
    keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
    keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(descriptors_train, descriptors_query)
    matches = sorted(matches, key = lambda x : x.distance)
    good = []
    bad = []
    for m in matches:
        if m.distance < 40:
            good.append([m])
        else:
            bad.append([m])
    if len(good) < 2 or len(bad) > 20:
        print("No matching points")
        sys.exit()
    src_pts = np.float32([ keypoints_train[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_query[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    return src_pts, dst_pts

def find_coordinates(src_pts, dst_pts, train_img):
    image = cv2.imread(train_img)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
    matchesMask = mask.ravel().tolist()
    h,w = image.shape[:2]
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    return dst

training_gray, query_gray = preprocess(sys.argv[1], sys.argv[2])
src_pts, dst_pts = find_match_pts(training_gray, query_gray)
dst = find_coordinates(src_pts, dst_pts,sys.argv[1])
print(dst)