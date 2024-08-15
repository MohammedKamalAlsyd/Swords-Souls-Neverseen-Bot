import cv2 as cv
import numpy as np


class HsvValues:
    hue_min = None
    saturation_min = None
    value_min = None
    hue_max = None
    saturation_max = None
    value_max = None
    saturation_increase = None
    saturation_decrease = None
    value_increase = None
    value_decrease = None


    def __init__(self, hue_min=None, saturation_min=None, value_min=None,
                 hue_max=None, saturation_max=None, value_max=None,
                 saturation_increase=None, saturation_decrease=None,
                 value_increase=None, value_decrease=None):
        self.hue_min = hue_min
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.hue_max = hue_max
        self.saturation_max = saturation_max
        self.value_max = value_max
        self.saturation_increase = saturation_increase
        self.saturation_decrease = saturation_decrease
        self.value_increase = value_increase
        self.value_decrease = value_decrease


class HsvFilter:
    trackbar_window_name = "HSV Filter"

    def getHSVValues(self):
        hsv_values = HsvValues(
            hue_min=cv.getTrackbarPos('Hue Min', self.trackbar_window_name),
            saturation_min=cv.getTrackbarPos('Saturation Min', self.trackbar_window_name),
            value_min=cv.getTrackbarPos('Value Min', self.trackbar_window_name),
            hue_max=cv.getTrackbarPos('Hue Max', self.trackbar_window_name),
            saturation_max=cv.getTrackbarPos('Saturation Max', self.trackbar_window_name),
            value_max=cv.getTrackbarPos('Value Max', self.trackbar_window_name),
            saturation_increase=cv.getTrackbarPos('Saturation Increase', self.trackbar_window_name),
            saturation_decrease=cv.getTrackbarPos('Saturation Decrease', self.trackbar_window_name),
            value_increase=cv.getTrackbarPos('Value Increase', self.trackbar_window_name),
            value_decrease=cv.getTrackbarPos('Value Decrease', self.trackbar_window_name)
        )
        return hsv_values

    def initControlGUI(self):
        cv.namedWindow(self.trackbar_window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.trackbar_window_name, 450, 700)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass


        # create trackbars for bracketing.
        # OpenCV scale for HSV is H: 0-255, S: 0-255, V: 0-255
        cv.createTrackbar('Hue Min', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Saturation Min', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Value Min', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Hue Max', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Saturation Max', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Value Max', self.trackbar_window_name, 0, 255, nothing)


        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('Saturation Increase', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Saturation Decrease', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Value Increase', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('Value Decrease', self.trackbar_window_name, 0, 255, nothing)

        # given an image and an HSV filter, apply the filter and return the resulting image.
        # if a filter is not supplied, the control GUI trackbars will be used
    def applyHSVFilter(self, original_image, hsv_filter=None):
        # convert image to HSV
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # if we haven't been given a defined filter, use the filter values from the GUI
        if not hsv_filter:
            hsv_filter = self.getHSVValues()

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    def shiftControl(self, c, amount):
        lim = abs(amount)
        c = np.clip(c + amount, 0, 255)
        return c