import sys
import numpy as np
import cv2
import glob

REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048

if len(sys.argv) != 2:
    print("Syntax: {0} CALIBRATION_FILE".format(sys.argv[0]))
    sys.exit(1)

calibration = np.load(sys.argv[1], allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

# The distortion in the left and right edges prevents a good calibration, so
# discard the edges
CROP_WIDTH = 960


def cropHorizontal(image):
    CAMERA_WIDTH = image.shape[1]
    return image[:,
           int((CAMERA_WIDTH - CROP_WIDTH) / 2):
           int(CROP_WIDTH + (CAMERA_WIDTH - CROP_WIDTH) / 2)]


# TODO: Why these values in particular?
# TODO: Try applying brightness/contrast/gamma adjustments to the images
stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)

files_left = glob.glob('capture/left/*.jpg')
files_right = glob.glob('capture/right/*.jpg')
images_left = [cv2.imread(img) for img in files_left]
images_right = [cv2.imread(img) for img in files_right]
assert len(images_left) == len(images_right), 'Number of left/right images does not match!'
images = [pair for pair in zip(images_left, images_right)]

# Grab both frames first, then retrieve to minimize latency between cameras
for leftFrame, rightFrame in images:

    leftFrame = cropHorizontal(leftFrame)
    leftHeight, leftWidth = leftFrame.shape[:2]
    rightFrame = cropHorizontal(rightFrame)
    rightHeight, rightWidth = rightFrame.shape[:2]

    if (leftWidth, leftHeight) != imageSize:
        raise IOError("Left camera has different size than the calibration data")

    if (rightWidth, rightHeight) != imageSize:
        raise IOError("Right camera has different size than the calibration data")

    fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
    fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

    grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
    grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
    depth = stereoMatcher.compute(grayLeft, grayRight)

    result = np.hstack([leftFrame, fixedLeft])

    scale_percent = 50  # percent of original size
    width = int(result.shape[1] * scale_percent / 100)
    height = int(result.shape[0] * scale_percent / 100)
    dim = (width, height)
    #result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

    #cv2.imshow('left org/fixed', result)
    cv2.imshow('left org/fixed', fixedLeft)

    # cv2.imshow('right', fixedRight)
    # cv2.imshow('depth', depth / DEPTH_VISUALIZATION_SCALE)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
