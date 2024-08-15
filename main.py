from Vision import vision
from ScreenshotCapture import Screen
from time import time
import pyautogui
import numpy as np
import cv2 as cv




if __name__ == "__main__":
    try: scr = Screen("Swords & Souls Neverseen", 40)
    except: scr = Screen()

    w, h = scr.getWindowDim()
    vis = vision("Data/Templates/Target.png", w, h)

    loop_time = time()

    while True:
        # Processing the image
        img = scr.capture()
        img = img[int(h//2.2):h, w//4:(3*w)//4]
        rectangles, _ = vis.find(img,0.45)

        img = vision.drawRectangles(img, rectangles)
        if 1 <= rectangles.shape[0] < 3:
            # Getting Missing Rectangles
            keys = {'a', 'w', 'd'}
            keys_copy = keys.copy()
            for rec in rectangles:
                x, y = rec[0], rec[1]
                print(x, y)
                if x >= img.shape[1] // 2:
                    keys_copy.discard('d')
                if y >= img.shape[0] // 3:
                    keys_copy.discard('w')
                if x <= img.shape[1] / 2:
                    keys_copy.discard('a')
            for k in keys_copy:
                pyautogui.press(k)






        cv.imshow("Test Screen", img)
        # Calculate and display the FPS
        fps = 1 / (time() - loop_time)
        #if True: print('FPS: {:.2f}'.format(fps))
        loop_time = time()

        # Press 'q' with the output window focused to exit
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break

    print('Done.')
