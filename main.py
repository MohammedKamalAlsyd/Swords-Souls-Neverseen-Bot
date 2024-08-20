from Vision import vision
from WindowCapture import Screen
from HSVFilter import HsvValues
from time import time
from Bot import Bot
import cv2 as cv

if __name__ == "__main__":
    # Initialize Classes
    scr = Screen("Swords & Souls Neverseen", 40)
    w, h = scr.getWindowDim()
    vis = vision("Data/Templates/Target.png", w, h)
    hsv_values = HsvValues(0, 0, 100, 179, 132, 255, 0, 0, 50, 0)  # Best Filter After Debugging
    game_bot = Bot(10, 300)

    # Initialize Some Variables
    loop_time = time()
    debug_mode = False

    # Starting Threads / HSV Filter GUI if needed
    scr.start()
    vis.start(0.65, debug_mode)
    game_bot.start()

    while True:
        # Processing the image
        img = scr.getCurrScreen()
        if img is None: continue
        img = img[int(h // 2.2):h, w // 4:(3 * w) // 4]

        # Finding Rectangles
        vis.update(img, hsv_values)
        rectangles = vis.getRectangles()
        if debug_mode:
            img = vision.drawRectangles(img, rectangles)
            cv.imshow("Debug Screen", img)

        # Passing Rectangles to the Bot
        game_bot.update(rectangles)

        # Calculate and display the FPS
        if debug_mode:
            fps = 1 / (time() - loop_time)
            print('FPS: {:.2f}'.format(fps))
        loop_time = time()

        # Press 'q' with the output window focused to exit
        if cv.waitKey(1) == ord('q'):
            scr.stop()
            vis.stop()
            game_bot.stop()
            cv.destroyAllWindows()
            break

    print('Done.')
