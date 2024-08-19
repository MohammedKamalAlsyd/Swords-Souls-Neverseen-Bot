from threading import Thread,Lock
import pyautogui

class bot:
    # Threading Properties
    _stopped = True # To Stop Capture if Stopped
    _lock = None # Act as Semaphore
    # Class Properties
    _curr_rectangles = [] # store x dim sine it unique across all different three rectangles

    _original_rectangles = []
    _pixels_treshold = None

    def __init__(self,pixels_treshold = 10):
        self._lock = Lock()
        self._pixels_treshold = pixels_treshold

    def update(self,rec):
        self._lock.acquire()
        self._curr_rectangles = rec
        self._lock.release()
        if len(self._original_rectangles) < 3:
            self._update_original(rec)


    def _update_original(self,rec):
        for r in rec:
            x = r[0]
            # Check if the list is empty or if the new x exceeds the threshold of pixels
            if all(abs(x - existing_rect[0]) > self._pixels_treshold for existing_rect in self._original_rectangles):
                self._original_rectangles.append(r)

        # Sort the _original_rectangles list by the x values
        self._original_rectangles.sort(key=lambda rect: rect[0])

    def _find_missing_indices(self, rect_list):
        missing_indices = []
        for i, stored_rect in enumerate(self._original_rectangles):
            found = False
            for rect in rect_list:
                if abs(rect[0] - stored_rect[0]) <= self._pixels_treshold:
                    found = True
                    break
            if not found:
                missing_indices.append(i + 1)  # Adding 1 to make it 1-based index
        return missing_indices


    def start(self):
        self._stopped = False
        t = Thread(target=self._run)
        t.start()


    def _run(self):
        while not self._stopped:
            if self._curr_rectangles is not None and len(self._curr_rectangles) >= 1:
                # get missing indicies
                missing = self._find_missing_indices(self._curr_rectangles)
                # Taking Actions
                if 0 in missing:
                    pyautogui.press('a')
                if 1 in missing:
                    pyautogui.press('w')
                if 2 in missing:
                    pyautogui.press('d')

    def stop(self):
        self._stopped = True