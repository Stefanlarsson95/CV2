import cv2
import threading
import time
import numpy as np
from multiprocessing import Lock, Value, Process
from multiprocessing.sharedctypes import RawArray
from threading import Thread, Event
from csi_camera import CSI_Camera
from stereovision.calibration import *
from stereovision.stereo_cameras import *


# from stereovision.blockmatchers import StereoBM, StereoSGBM


class CameraHandler:
    def __init__(self, display_fps=False, rotate=0, scale=1):
        """
        Camera handler utilizing multiprocessing to obtain frames from multiple cameras in on process and
        obtaining DepthMap in another.
        :param image_size: Size of camera capture, by default 1280x720 as natively by RPi CSI cameras
        :param display_fps: Display framerate in frame
        :param rotate: rotate image of each camera
        """

        # Initialize instance variables
        self._n_cameras = 2
        self._display_fps = display_fps
        self._rotate = rotate
        self._scale = scale
        self._image_size_camera = [720, 1280, 3]
        self._image_size_stereo = self._image_size_camera * np.transpose([1, 1, self._n_cameras])
        self._image_size_depth = self._image_size_camera[0:2]
        self.n_pixels = int(np.prod(self._image_size_camera) * scale)
        self.mp_frame_lock = Lock()
        self.mp_depth_lock = Lock()
        self.frame_lock = Lock()
        self.depth_lock = Lock()
        self.mp_raw_image = RawArray('d', self.n_pixels * self._n_cameras)
        self.mp_raw_depth_image = RawArray('d', int(self.n_pixels / 3))
        self._running_flag = Value('i', 0)
        self._t_frame = Value('f', 0.0)
        self._t_depth_frame = Value('f', 0.0)

        # camera vars
        self.calib_file = 'calibration'
        self.block_matcher = 0  # StereoSGBM()  # alt StereoBM() Todo fix
        self.calibration = 0  # StereoCalibration(input_folder=self.calib_file)

        # globals
        self.running = False
        self.color_frame = np.zeros(self._image_size_stereo).astype(np.uint8)
        self.depth_frame = np.zeros(self._image_size_depth).astype(np.uint8)

        # frame handlers
        self._color_frame_capture_process = Process(target=self._camera_handler, name="CameraHandlerProcess", )
        self._depth_frame_generator = Process(target=self._depth_handler, name="DepthMapProcess", )
        self._frame_handler_thread = Thread(target=self._frame_handler, name="FramePollingThread")
        self._depth_frame_handler_thread = Thread(target=self._frame_handler, name="DepthMapGeneratorThread",
                                                  args=(True,))

    def star(self, depth_handler=False):
        if not self.running:
            self.running = True
        else:
            print('CameraHandler already running!')
            return None

        self._color_frame_capture_process.start()
        self._frame_handler_thread.start()
        if depth_handler:
            self._depth_frame_generator.start()
            self._depth_frame_handler_thread.start()
        return self

    def stop(self):
        with self._running_flag.get_lock():
            if self._running_flag.value == 1:
                self._running_flag.value = 0
            else:
                return False
        if self._color_frame_capture_process.is_alive():
            self._color_frame_capture_process.join()
        if self._depth_frame_generator.is_alive():
            self._depth_frame_generator.join()
        if self._frame_handler_thread.is_alive():
            self._frame_handler_thread.join()
        if self._depth_frame_handler_thread.is_alive():
            self._depth_frame_handler_thread.join()
        return True

    def get_frame(self, depth_frame=False):
        """
        Captured images
        :param depth_frame: if return depth data as grayscale image
        :return: dual rgb image or grayscale depth image
        """

        if depth_frame:
            if not self._depth_frame_handler_thread.is_alive():
                self._depth_frame_handler_thread.start()
            with self.depth_lock:
                img = self.depth_frame
        else:
            with self.frame_lock:
                img = self.color_frame

        return img

    def get_running_state(self):
        with self._running_flag.get_lock():
            return self._running_flag.value == 1

    def _frame_handler(self, depth_frame=False):
        t_last_frame = 0
        self.running = True
        _attempts = 0
        with self._running_flag.get_lock():
            self._running_flag.value = 1

        while _attempts <= 100:
            with self._running_flag.get_lock():
                self.running = self._running_flag.value == 1
            if not self.running:
                break

            if not depth_frame:
                with self._t_frame.get_lock():
                    t_latest_frame = self._t_frame.value
                if t_latest_frame - t_last_frame < 0.005:  # capt to <200fps
                    time.sleep(0.1)
                    _attempts += 1
                    continue
                with self.mp_frame_lock:
                    size = self._image_size_camera
                    size[2] = 6  # 1280*2  # set size for dual frame
                    img = self.rawarray_to_nparray(self.mp_raw_image, image_size=size)
                with self.frame_lock:
                    self.color_frame = img
            else:
                with self.mp_depth_lock:
                    img = self.rawarray_to_nparray(self.mp_raw_depth_image, self._image_size_depth)
                with self.depth_lock:
                    self.depth_frame = img

        print('Frame handler ended')

    def _camera_handler(self):

        # initiate cameras
        camera_0 = CSI_Camera(self._image_size_camera)
        camera_1 = CSI_Camera(self._image_size_camera)
        cameras = [camera_0, camera_1]
        for sensor_id, camera in enumerate(cameras):
            camera.create_gstreamer_pipeline(sensor_id=sensor_id)
            camera.open()
            camera.display_fps = self._display_fps
            camera.scale = 1
            camera.start()

        _attempts = 0
        while _attempts <= 10:
            # check for termination signal
            with self._running_flag.get_lock():
                if self._running_flag.value != 1:
                    break

            # check frame update for all cameras
            frames = []
            for camera in cameras:
                if camera.frame_id != camera.last_frame_id:
                    camera.last_frame_id = camera.frame_id
                    grabbed, frame = camera.read_frame()
                    if grabbed:
                        if self._rotate:
                            frame = cv2.rotate(frame, self._rotate)
                        frames.append(frame)
                    else:
                        _attempts += 1
                        continue
                time.sleep(0.01)
            if len(frames) < self._n_cameras:
                _attempts += 1
                continue
            np_frames = np.dstack(frames).astype(np.uint8)
            # update raw image array
            with self.mp_frame_lock:
                self.nparray_to_rawarray(np_frames, self.mp_raw_image)

            # update image timestamp
            with self._t_frame.get_lock():
                self._t_frame.value = time.perf_counter()

            time.sleep(0)

        for camera in cameras:
            camera.stop()
            camera.release()

    def _depth_handler(self):
        """
        get depth image from stereo frames
        from example: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html
        :return: grayscale DepthMap as RawArray
        """
        last_frame_time = 0
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        while True:
            with self._running_flag.get_lock():
                if self._running_flag.value != 1:
                    break
            with self._t_frame.get_lock():
                frame_time = self._t_frame.value

            if frame_time != last_frame_time:
                last_frame_time = frame_time

                with self.frame_lock:
                    raw_frame = self.mp_raw_image

                size = self._image_size_camera
                size[2] = 6  # 1280*2  # set size for dual frame
                dual_frames = self.rawarray_to_nparray(raw_frame, image_size=size)

                frame_left, frame_right = np.split(dual_frames, indices_or_sections=2, axis=2)

                # todo check if necessary?
                gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

                # rectify frames from warp and disalignment
                # rectified_pair = calibration.rectify([gray_left, gray_right])
                rectified_pair = self.calibration.rectify(dual_frames)

                # disparity = stereo.compute(gray_left, gray_right)
                disparity = self.block_matcher.get_disparity(rectified_pair)

                with self.mp_depth_lock:
                    self.nparray_to_rawarray(disparity, self.mp_raw_depth_image)

                with self._t_depth_frame.get_lock():
                    self._t_depth_frame.value = time.perf_counter()
                time.sleep(1 / 50)
            else:
                time.sleep(1)

    def rawarray_to_nparray(self, raw_array, image_size=None):
        if not image_size:
            image_size = self._image_size_camera
        return np.frombuffer(raw_array).reshape(image_size).astype(np.uint8)

    @staticmethod
    def nparray_to_rawarray(arr, raw_array):
        np.frombuffer(raw_array).reshape(arr.shape)[...] = arr


def mp_test():
    mp_camera = CameraHandler(display_fps=True, rotate=0).star()
    print('Starting mp camera test..')
    time.sleep(1)
    dual_frame = mp_camera.get_frame()
    left_image = dual_frame[:, :, 0:3]
    right_image = dual_frame[:, :, 3:]

    try:
        cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
        # Start counting the number of frames read and displayed
        while cv2.getWindowProperty("CSI Cameras", 0) >= 0 and mp_camera.get_running_state():
            t_start = time.perf_counter()
            dual_frame = mp_camera.get_frame()
            left_image = dual_frame[:, :, 0:3]
            right_image = dual_frame[:, :, 3:]
            if not left_image.shape == right_image.shape:
                time.sleep(0.1)
                continue

            # We place both images side by side to show in the window
            camera_images = np.hstack((left_image, right_image))
            # camera_images = left_image
            scale_percent = 75  # percent of original size
            width = int(camera_images.shape[1] * scale_percent / 100)
            height = int(camera_images.shape[0] * scale_percent / 100)
            dim = (width, height)
            camera_images = cv2.resize(camera_images, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("CSI Cameras", camera_images)
            # This also acts as a frame limiter
            # Stop the program on the ESC key
            if (cv2.waitKey(20) & 0xFF) == 27:
                break
            t_end = time.perf_counter()
            t_elapse = t_end - t_start
            time.sleep(max(1 / 30 - t_elapse, 0))
    except KeyboardInterrupt:
        pass
    finally:
        mp_camera.stop()
    cv2.destroyAllWindows()


def mp_test_depth():
    mp_camera = CameraHandler(display_fps=True, rotate=0).star(depth_handler=True)
    print('Starting mp depth frame test..')
    time.sleep(1)
    depth_frame = mp_camera.get_frame(depth_frame=True)
    time.sleep(10)
    try:
        cv2.namedWindow("CSI Cameras", cv2.WINDOW_AUTOSIZE)
        t_last_frame = 0
        # Start counting the number of frames read and displayed
        while cv2.getWindowProperty("CSI Cameras", 0) >= 0 and mp_camera.get_running_state():
            t_start = time.perf_counter()
            depth_frame = mp_camera.get_frame(depth_frame=True)
            print(depth_frame)
            scale_percent = 75  # percent of original size
            width = int(depth_frame.shape[1] * scale_percent / 100)
            height = int(depth_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            depth_frame = cv2.resize(depth_frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("CSI Cameras", depth_frame)
            # This also acts as a frame limiter
            # Stop the program on the ESC key
            if (cv2.waitKey(20) & 0xFF) == 27:
                break
            t_end = time.perf_counter()
            t_elapse = t_end - t_start
            time.sleep(max(1 / 30 - t_elapse, 0))
    except KeyboardInterrupt:
        pass
    finally:
        mp_camera.stop()
    cv2.destroyAllWindows()


def mp_test_no_screen():
    mp_camera = CameraHandler(display_fps=True, rotate=0).star()
    print('Starting mp camera test..')
    time.sleep(1)

    try:
        # Start counting the number of frames read and displayed
        while True:
            dual_frame = mp_camera.get_frame()
            left_image = dual_frame[:, :, 0:3]
            right_image = dual_frame[:, :, 3:]
            print(left_image)
            print(right_image)
            time.sleep(1 / 5)
    except KeyboardInterrupt:
        pass
    finally:
        mp_camera.stop()


def mp_depth_test_no_screen():
    mp_camera = CameraHandler(display_fps=True, rotate=0).star(depth_handler=True)

    print('Starting mp camera test..')
    time.sleep(1)

    try:
        # Start counting the number of frames read and displayed
        while True:
            depth_frame = mp_camera.get_frame(depth_frame=True)
            print(depth_frame)
            time.sleep(1 / 5)
    except KeyboardInterrupt:
        pass
    finally:
        mp_camera.stop()


def depth_test():
    from matplotlib import pyplot as plt
    imgL = cv2.imread('left.jpg', 0)
    imgR = cv2.imread('right.jpg', 0)
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    print(disparity)
    plt.imshow(disparity, 'gray')
    plt.show()


def capture_stereo(save_folder='capture', interval=3, n_images=30):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder + '/left')
        os.makedirs(save_folder + '/right')
        os.makedirs(save_folder + '/combined')

    # Different directories for each camera
    LEFT_PATH = save_folder + "/left/left" + "{:06d}.jpg"
    RIGHT_PATH = save_folder + "/right/right" + "{:06d}.jpg"
    COMBINED_PATH = save_folder + "/combined/combined" + "{:06d}.jpg"

    mp_camera = CameraHandler(rotate=0).star()
    print('Starting mp camera test..')
    time.sleep(1)

    # Filenames are just an increasing number
    frameId = 0
    countdown = interval
    try:
        cv2.namedWindow("Capture stereo images", cv2.WINDOW_AUTOSIZE)
        # Start counting the number of frames read and displayed
        t_last_image = time.perf_counter()
        while cv2.getWindowProperty("Capture stereo images", 0) >= 0 and mp_camera.get_running_state():
            t_start = time.perf_counter()
            dual_frame = mp_camera.get_frame()
            left_image = dual_frame[:, :, 0:3]
            right_image = dual_frame[:, :, 3:]
            if not left_image.shape == right_image.shape:
                time.sleep(0.1)
                continue

            dt = time.perf_counter() - t_last_image
            if countdown <= 0:
                # Actually save the frames
                cv2.imwrite(LEFT_PATH.format(frameId), left_image)
                cv2.imwrite(RIGHT_PATH.format(frameId), right_image)
                cv2.imwrite(COMBINED_PATH.format(frameId), dual_frame[0])
                frameId += 1
                countdown = interval
                t_last_image = time.perf_counter()
            elif np.ceil(interval - dt) <= countdown:
                countdown -= 1

            # We place both images side by side to show in the window
            combined_image = np.hstack((left_image, right_image))

            cv2.putText(combined_image, ('Countdown: {}'.format(countdown)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined_image, ('Frame count: {}/{}'.format(frameId, n_images)), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # camera_images = left_image
            scale_percent = 75  # percent of original size
            width = int(combined_image.shape[1] * scale_percent / 100)
            height = int(combined_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            combined_image = cv2.resize(combined_image, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Capture stereo images", combined_image)
            if frameId == n_images:
                time.sleep(1)
                break
            # This also acts as a frame limiter
            # Stop the program on the ESC key
            if (cv2.waitKey(20) & 0xFF) == 27:
                break
            t_end = time.perf_counter()
            t_elapse = t_end - t_start
            time.sleep(max(1 / 30 - t_elapse, 0))
    finally:
        mp_camera.stop()
        cv2.destroyAllWindows()


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
        '/home/unicornnx/descento/colcon_ws/src/hmi_io/hmi_io/capture/right/right*')  # Make a list of paths to calibration images
    # Step through the list and search for chessboard corners
    img = corners = ret = []
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


def generate_calibrator(indir='capture', outdir='stereo_calibration', rows=9, columns=9, square_size=1.8):
    import glob

    filenames = glob.glob(indir + "/*.jpg")
    if not filenames:
        raise FileNotFoundError('No images found!')
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]

    image_size = np.size(images[0])

    # create calibrator object
    calibrator = StereoCalibrator(rows, columns, square_size, image_size)

    # add image pairs to calibrator
    for img in images:
        calibrator.add_corners((img[0], img[1]))
    # run calibrator
    calibration = calibrator.calibrate_cameras()

    avg_error = calibrator.check_calibration(calibration)

    print('Avg error: {}'.format(avg_error))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    calibration.export(outdir)


if __name__ == '__main__':
    # mp_test()
    # mp_test_no_screen()
    # mp_depth_test_no_screen()
    # mp_test_depth()
    # depth_test()
    # capture_stereo()
    show_corners()
