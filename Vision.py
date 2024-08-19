import cv2 as cv
import numpy as np
from threading import Thread, Lock
from HSVFilter import HsvFilter, HsvValues

class vision:
    # Threading Properties
    _stopped = True # To Stop Capture if Stopped
    _lock = None # Act as Semaphore
    _rectangles = np.array([], dtype=np.int32).reshape(0, 4)
    _screenshot = None
    # Class Properties
    _hsv_filter = None
    template = None
    template_width = None
    template_height = None
    method = cv.TM_CCOEFF_NORMED # You Should Try Different Methods. Anyway I test all and this was the best
    # There are 6 methods to choose from:
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED


    def _calculate_rescale_percentage(self,new_width, new_height,original_width=720, original_height=400): # original_width & original_width from Cropped template original dim
        width_scale = new_width / original_width
        height_scale = new_height / original_height
        return width_scale, height_scale


    def __init__(self,template_path,new_width,new_height,method = None):
        self._hsv_filter = HsvFilter()
        self._lock = Lock()
        # Transform Template Size to work on different screen sizes
        try:
            self.template = cv.imread(template_path,cv.IMREAD_UNCHANGED)
        except:
            print(f"Failed to read {template_path}")
            exit(0)
        width_scale, height_scale = self._calculate_rescale_percentage(new_width,new_height)  # Need to get Curr Windows Size
        self.template = cv.resize(self.template, None, fx=width_scale, fy=height_scale, interpolation=cv.INTER_LINEAR)
        self.template_height = self.template.shape[0]
        self.template_width = self.template.shape[1]
        if method: self.method = method
        if self.template.shape[-1] == 4: self.template = cv.cvtColor(self.template,cv.COLOR_BGRA2BGR)


    def find(self,img,threshold=0.5,group_treshold=1,eps=0.1,max_results=100):
        Similarity = cv.matchTemplate(img, self.template,self.method) # cv.TM_CCOEFF_NORMED

        # Getting Matches
        top_left_corners = np.where(Similarity >= threshold)
        top_left_corners = list(zip(*top_left_corners[::-1]))

        # No Match Found
        if not top_left_corners:
            return np.array([], dtype=np.int32).reshape(0, 4), None # 4 representing (x, y, w, h)

        # Formulate Rectangles
        rectangles = [(rec[0], rec[1], self.template_width, self.template_height) for rec in top_left_corners]
        rectangles, weights = cv.groupRectangles(rectangles, group_treshold, eps)
        if len(rectangles) > max_results:
            print('Warning: too many results, raise the threshold.')
            rectangles = rectangles[:max_results]

        if isinstance(rectangles, tuple):
            return np.array([], dtype=np.int32).reshape(0, 4), None # 4 representing (x, y, w, h)


        return rectangles, weights


    def start(self,threshold, debug_mode):
        self._debug_mode = debug_mode
        if debug_mode: self._hsv_filter.initControlGUI()
        self._stopped = False
        t = Thread(target=self._run, args=(threshold,))
        t.start()


    def stop(self):
        self._stopped = True


    def _run(self,threshold):
        while not self._stopped:
            if self._screenshot is not None:
                # get an updated rectangles of the img
                rectangles, _ = self.find(self._screenshot,threshold)
                # lock the thread while updating the results
                self._lock.acquire()
                self._rectangles = rectangles
                self._lock.release()


    def update(self, screenshot, hsv_values = None):
        self._lock.acquire()
        if self._debug_mode: self._screenshot = self._hsv_filter.applyHSVFilter(screenshot)
        else: self._screenshot = self._hsv_filter.applyHSVFilter(screenshot, hsv_values)  # Apply Best Values
        self._lock.release()


    def getRectangles(self):
        return self._rectangles


    def getScreenshot(self):
        return self._screenshot


    @staticmethod
    def drawRectangles(img,rectangles,line_color = (255, 0, 0),line_type = cv.LINE_4,thickness=2):
        for (x, y, w, h) in rectangles:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # Draw the box
            cv.rectangle(img, top_left, bottom_right, color=line_color, lineType=line_type, thickness=thickness)
        return img

    @staticmethod
    def draw_crosshairs(img, points,marker_color = (255, 0, 255),marker_type = cv.MARKER_CROSS):
        for (center_x, center_y) in points: cv.drawMarker(img, (center_x, center_y), marker_color, marker_type)
        return img