import win32gui
from time import time
from WindowCapture import Screen
import cv2 as cv



# Initialize the loop for capturing screenshots
try:
    # max fps not very accurate. anyway 40 capture from 30 to 34
    # Note Also it Failed to Capture Browsers
    scr = Screen("Swords & Souls Neverseen",40)
except:
    print("Windows Not Found")
    exit(0)

loop_time = time()
while True:
    img = scr.capture()
    print(img.shape)
    cv.imshow("t",img)
    # Calculate and display the FPS
    fps = 1 / (time() - loop_time)
    print('FPS: {:.2f}'.format(fps))
    loop_time = time()



    # Press 'q' with the output window focused to exit
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
