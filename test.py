from WindowCapture import Screen
from time import time
import numpy as np
import cv2 as cv

if __name__ == "__main__":
    try:
        scr = Screen("Swords & Souls Neverseen", 40)
        w, h = scr.getWindowDim()
    except:
        print("Window Not Found")
        exit(0)

    # Initialize the background subtractor
    back_sub = cv.createBackgroundSubtractorMOG2()

    loop_time = time()
    while True:
        img = scr.capture()

        # Convert the captured image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply background subtraction to get the foreground mask
        fg_mask = back_sub.apply(gray)

        # Find contours of the moving objects in the mask
        contours, _ = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected moving objects
        for contour in contours:
            if cv.contourArea(contour) > 500:  # Filter out small contours
                (x, y, w, h) = cv.boundingRect(contour)
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the foreground mask
        cv.imshow("Foreground Mask", fg_mask)

        # Display the current frame with detected moving objects
        cv.imshow("Motion Detected", img)

        # Calculate and display the FPS
        fps = 1 / (time() - loop_time)
        print('FPS: {:.2f}'.format(fps))
        loop_time = time()

        # Press 'q' with the output window focused to exit
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break

    print('Done.')
