import win32gui
import win32ui
import win32con
import numpy as np
import cv2 as cv
import time
from threading import Thread, Lock


class Screen:
    # Threading Properties
    _stopped = True # To Stop Capture if Stopped
    _lock = None # Act as Semaphore
    _screenshot = None
    # Class Properties
    _window_name = None
    _hwnd = None
    _h = 1080
    _w = 1920
    ## Top Left Axes
    local_TL_x = 0
    local_TL_y = 0
    ## Position After Cropping
    global_TL_x = 0
    global_TL_y = 0


    def __init__(self,window_name=None,borders_pixels = 8,titlebar_pixels=31):
        self._window_name = window_name
        self._lock = Lock()

        if self._window_name:
            self._hwnd = win32gui.FindWindow(None, self._window_name)
            if not self._hwnd:
                print("Window Not Found. Please Check The Name")
                exit(0)
        else:
            self._hwnd = win32gui.GetDesktopWindow()

        rect = win32gui.GetWindowRect(self._hwnd) # (TL_x,TL_y,width,height)
        self._w = rect[2] - rect[0]
        self._h = rect[3] - rect[1]


        # apply some croping as pywin32 capture the whole windows including the borders
        if self._window_name:
            self._w = self._w - (borders_pixels * 2)
            self._h = self._h - titlebar_pixels - borders_pixels
            self.local_TL_x = borders_pixels
            self.local_TL_y = titlebar_pixels

        # Actual Positions
        self.global_TL_x = rect[0] + self.local_TL_x
        self.global_TL_y = rect[1] + self.local_TL_y



    def getWindowDim(self):
        return self._w,self._h


    def get_screen_position(self, pos):
        return (pos[0] + self.global_TL_x, pos[1] + self.global_TL_y)


    def _capture(self):
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
        cDC.BitBlt((0, 0), (self._w, self._h), dcObj, (self.local_TL_x, self.local_TL_y), win32con.SRCCOPY)

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

        return img


    def start(self):
        self._stopped=False
        t = Thread(target=self._run)
        t.start()


    def stop(self):
        self._stopped = True


    def _run(self):
        while not self._stopped:
            # get an updated image of the game
            screenshot = self._capture()
            # lock the thread while updating the results
            self._lock.acquire()
            self._screenshot = screenshot
            self._lock.release()


    def getCurrScreen(self):
        return self._screenshot


    @staticmethod
    def listWindowsNames():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(winEnumHandler, None)