import cv2 as cv
import numpy as np

class vision:
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
        # Transform Template Size to work on different screen sizes
        self.template = cv.imread(template_path,cv.IMREAD_UNCHANGED)
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

        return rectangles, weights

    @staticmethod
    def drawRectangles(img,rectangles,line_color = (0, 255, 0),line_type = cv.LINE_4,thickness=2):
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


