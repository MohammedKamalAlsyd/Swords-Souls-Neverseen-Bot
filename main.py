from Vision import vision
from WindowCapture import Screen
from HSVFilter import HsvFilter, HsvValues
from time import time,sleep
import pyautogui
import numpy as np
import cv2 as cv



if __name__ == "__main__":
    # Initialize Classes
    scr = Screen("Swords & Souls Neverseen", 40)
    w, h = scr.getWindowDim()
    vis = vision("Data/Templates/Target2.png", w, h)
    hsv_values = HsvValues(0, 0, 100, 179, 132, 255, 0, 0, 50, 0)  # Best Filter After Debugging


    # Initialize Some Variables
    loop_time = time()
    debug_mode = True


    # Starting Threads / HSV Filter GUI if needed
    scr.start()
    vis.start(0.7, debug_mode)



    while True:
        # Processing the image
        img = scr.getCurrScreen()
        if img is None: continue
        img = img[int(h//2.2):h, w//4:(3*w)//4]

        # Finding Rectangles
        vis.update(img,hsv_values)
        rectangles = vis.getRectangles()
        if debug_mode:
            img = vision.drawRectangles(img, rectangles)
            cv.imshow("Debug Screen", img)



    #     # if 1 <= rectangles.shape[0] < 3: #not isinstance(rectangles,tuple ) and
    #     #     # Getting Missing Rectangles
    #     #     keys = {'a', 'w', 'd'}
    #     #     keys_copy = keys.copy()
    #     #     for rec in rectangles:
    #     #         x, y = rec[0], rec[1]
    #     #         if x >= img.shape[1] // 2:
    #     #             keys_copy.discard('d')
    #     #         if y >= img.shape[0] // 3:
    #     #             keys_copy.discard('w')
    #     #         if x <= img.shape[1] / 2:
    #     #             keys_copy.discard('a')
    #     #     for k in keys_copy:
    #     #         pyautogui.press(k)
    #
    #
    #
    #
        # Calculate and display the FPS
        if debug_mode:
            fps = 1 / (time() - loop_time)
            # print('FPS: {:.2f}'.format(fps))
            # print(rectangles.shape)
        loop_time = time()
    #
        # Press 'q' with the output window focused to exit
        if cv.waitKey(1) == ord('q'):
            scr.stop()
            vis.stop()
            cv.destroyAllWindows()
            break

    print('Done.')
