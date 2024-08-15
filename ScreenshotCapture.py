import win32gui
import win32ui
import win32con
import numpy as np
import cv2 as cv
import time


class Screen:
    _window_name = None
    _hwnd = None
    _h = 1080
    _w = 1920
    _TL_x = 0
    _TL_y = 0


    def __init__(self,window_name=None,max_fps=30,borders_pixels = 8,titlebar_pixels=31):
        self._window_name = window_name
        self.max_fps = max_fps
        self.interval = 1.0 / max_fps  # Interval in seconds

        if self._window_name:
            self._hwnd = win32gui.FindWindow(None, self._window_name)
        else:
            self._hwnd = win32gui.GetDesktopWindow()

        rect = win32gui.GetWindowRect(self._hwnd)
        self._w = rect[2] - rect[0]
        self._h = rect[3] - rect[1]

        # apply some croping as pywin32 capture the whole windows including the borders
        if self._window_name:
            self._w = self._w - (borders_pixels * 2)
            self._h = self._h - titlebar_pixels - borders_pixels
            self._TL_x = borders_pixels
            self._TL_y = titlebar_pixels


    def getWindowDim(self):
        return self._w,self._h

    def capture(self):
        # To Capture 1 Frame Time
        start_time = time.time()

        # Get the window device context (DC)
        wDC = win32gui.GetWindowDC(self._hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()

        # Create a bitmap object
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self._w, self._h)
        cDC.SelectObject(dataBitMap)

        # Capture the screenshot
        cDC.BitBlt((0, 0), (self._w, self._h), dcObj, (self._TL_x, self._TL_y), win32con.SRCCOPY)

        # Convert the bitmap to a NumPy array for OpenCV
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self._h, self._w, 4)  # 4 channels: BGRA

        # Convert the Img to CV format
        # img = np.ascontiguousarray(img)
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

        # Free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self._hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # Enforce the maximum FPS
        elapsed_time = time.time() - start_time
        sleep_time = self.interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        return img

    @staticmethod
    def listWindowsNames():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(winEnumHandler, None)

