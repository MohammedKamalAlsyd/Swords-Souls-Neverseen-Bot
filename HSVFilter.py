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
            hue_min=cv.getTrackbarPos('H Min', self.trackbar_window_name),
            saturation_min=cv.getTrackbarPos('S Min', self.trackbar_window_name),
            value_min=cv.getTrackbarPos('V Min', self.trackbar_window_name),
            hue_max=cv.getTrackbarPos('H Max', self.trackbar_window_name),
            saturation_max=cv.getTrackbarPos('S Max', self.trackbar_window_name),
            value_max=cv.getTrackbarPos('V Max', self.trackbar_window_name),
            saturation_increase=cv.getTrackbarPos('S Inc', self.trackbar_window_name),
            saturation_decrease=cv.getTrackbarPos('S Dec', self.trackbar_window_name),
            value_increase=cv.getTrackbarPos('V Inc', self.trackbar_window_name),
            value_decrease=cv.getTrackbarPos('V Dec', self.trackbar_window_name)
        )
        return hsv_values

    def initControlGUI(self):
        cv.namedWindow(self.trackbar_window_name, cv.WINDOW_AUTOSIZE)
        cv.resizeWindow(self.trackbar_window_name, 400, 380)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass


        # create trackbars for bracketing.
        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('H Min', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('S Min', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('V Min', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('H Max', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('S Max', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('V Max', self.trackbar_window_name, 0, 255, nothing)

        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('H Max', self.trackbar_window_name, 179)
        cv.setTrackbarPos('S Max', self.trackbar_window_name, 255)
        cv.setTrackbarPos('V Max', self.trackbar_window_name, 255)


        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('S Inc', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('S Dec', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('V Inc', self.trackbar_window_name, 0, 255, nothing)
        cv.createTrackbar('V Dec', self.trackbar_window_name, 0, 255, nothing)

        # given an image and an HSV filter, apply the filter and return the resulting image.
        # if a filter is not supplied, the control GUI trackbars will be used


    def applyHSVFilter(self,original_image, hsv_filter=None):
       # Function To Avoid Over Saturation (RGB Value Exceed 255)
        def shiftControl(c, amount):
            if amount > 0:
                lim = 255 - amount
                c[c >= lim] = 255
                c[c < lim] += amount
            elif amount < 0:
                amount = -amount
                lim = amount
                c[c <= lim] = 0
                c[c > lim] -= amount
            return c

        # convert image to HSV
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # if we haven't been given a defined filter, use the filter values from the GUI
        if not hsv_filter:
            hsv_filter = self.getHSVValues()

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = shiftControl(s, hsv_filter.saturation_increase)
        s = shiftControl(s, -hsv_filter.saturation_decrease)
        v = shiftControl(v, hsv_filter.value_increase)
        v = shiftControl(v, -hsv_filter.value_decrease)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsv_filter.hue_min, hsv_filter.saturation_min, hsv_filter.value_min])
        upper = np.array([hsv_filter.hue_max, hsv_filter.saturation_max, hsv_filter.value_max])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img