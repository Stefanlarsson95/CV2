import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from stereovision.blockmatchers import *
from stereovision.calibration import *
from stereovision.ui_utils import *

block_matcher = StereoSGBM()  # alt StereoBM() Todo fix
calibration = StereoCalibration(input_folder='old_stereo_calibration')


def plot_disparuty():
    imgL = cv2.imread('capture/left.jpg*', 0)
    imgR = cv2.imread('capture/right.jpg*', 0)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'gray')
    plt.show()


def show_corners():
    import glob
    Nx_cor = 9  # Number of corners to find
    Ny_cor = 6
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((Nx_cor * Ny_cor, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(
        'capture/right/right*')  # Make a list of paths to calibration images
    # Step through the list and search for chessboard corners
    img_w_corner = []
    corners_not_found = []  # Calibration images in which OpenCV failed to find corners
    print('Calculating corners')
    for idx, fname in enumerate(images):
        print('frame: {}'.format(idx))
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), None)  # Find the corners
        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (Nx_cor, Ny_cor), corners, True)
            img_w_corner.append(img)
        else:
            corners_not_found.append(fname)
    print('Calculation done!')
    cv2.namedWindow("Corners", cv2.WINDOW_AUTOSIZE)
    for img_pt in img_w_corner:
        # Draw corners
        cv2.imshow("Corners", cv2.rotate(img_pt, 2))
        if (cv2.waitKey(3000) & 0xFF) == 27:
            break
    cv2.destroyAllWindows()


def generate_calibrator(indir='capture', outdir='old_stereo_calibration', rows=6, columns=9, square_size=1.8):
    image_size = (720, 1280-500)
    print('Stating camera calibrator...')
    print('Importing files...')
    files_left = glob.glob(indir + '/left/*')
    # files_left.sort()
    files_right = glob.glob(indir + '/right/*')
    # files_right.sort()
    images_left = [cv2.imread(img)[:, 250:1280-250] for img in files_left]
    images_right = [cv2.imread(img)[:, 250:1280-250] for img in files_right]
    assert len(images_left) == len(images_right), 'Number of left/right images does not match!'

    images = [pair for pair in zip(images_left, images_right)]

    print('Creating image calibrator...')
    # create calibrator object
    calibrator = StereoCalibrator(rows, columns, square_size, image_size)

    # add image pairs to calibrator
    for i, img in enumerate(images):
        print('scanning image pair: {}'.format(i))
        gray_left = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img[1], cv2.COLOR_BGR2GRAY)
        ret_left, _ = cv2.findChessboardCorners(gray_left, (rows, columns), None)  # Find the corners
        ret_right, _ = cv2.findChessboardCorners(gray_right, (rows, columns), None)  # Find the corners
        if not (ret_left and ret_right):
            print('No chessboard found in image pair {}'.format(i))
            continue
        calibrator.add_corners((img[0], img[1]))
    # run calibrator
    print('Generating calibration...')
    calibration = calibrator.calibrate_cameras()

    print('Calculating average error: ', end='')
    avg_error = calibrator.check_calibration(calibration)
    print(avg_error)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    calibration.export(outdir)


def show_rectified(indir='capture'):
    files_left = glob.glob(indir + '/left/*')
    # files_left.sort()
    files_right = glob.glob(indir + '/right/*')
    # files_right.sort()
    images_left = [cv2.imread(img) for img in files_left]
    images_right = [cv2.imread(img) for img in files_right]
    assert len(images_left) == len(images_right), 'Number of left/right images does not match!'
    images = [pair for pair in zip(images_left, images_right)]

    cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)

    for pairs in images:
        rectified_pair = calibration.rectify(pairs)

        left = rectified_pair[0]
        right = rectified_pair[1]
        # left = cv2.rotate(left, 1)
        # right = cv2.rotate(right, 1)
        cv2.imshow("Calibration", np.hstack([left, right]))
        if (cv2.waitKey(3000) & 0xFF) == 27:
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()


def BMtune(indir='capture'):
    block_matcher = StereoSGBM()

    files_left = glob.glob(indir + '/left/*')
    # files_left.sort()
    files_right = glob.glob(indir + '/right/*')
    # files_right.sort()
    images_left = [cv2.imread(img) for img in files_left]
    images_right = [cv2.imread(img) for img in files_right]
    assert len(images_left) == len(images_right), 'Number of left/right images does not match!'
    images = [pair for pair in zip(images_left, images_right)]

    rectified_pair = calibration.rectify(images[10])
    cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)

    cv2.imshow("Calibration", cv2.rotate(np.vstack(rectified_pair), 2))
    if (cv2.waitKey(3000) & 0xFF) == 27:
        cv2.destroyAllWindows()
        return

    bm_tuner = BMTuner(block_matcher, calibration, rectified_pair)

    for pair in images:
        rectified_pair = calibration.rectify(pair)
        bm_tuner.tune_pair(rectified_pair)

    for param in block_matcher.parameter_maxima:
        print("{}\n".format(bm_tuner.report_settings(param)))


if __name__ == '__main__':
    generate_calibrator()
    # show_corners()
    show_rectified()
    # BMtune()
