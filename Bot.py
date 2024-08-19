from threading import Thread, Lock
import pyautogui
import time


class Bot:
    # Threading Properties
    _stopped = True  # To Stop Capture if Stopped
    _lock = None  # Act as Semaphore
    # Class Properties
    _curr_rectangles = []  # store x dim sine it unique across all different three rectangles
    _time_treshold_ms = None  # Time threshold in milliseconds
    _original_rectangles = []
    _pixels_treshold = None
    _last_press_time = {}  # Store last press time for each key

    def __init__(self, pixels_treshold=10, time_treshold_ms=100):
        self._lock = Lock()
        self._pixels_treshold = pixels_treshold
        self._time_treshold_ms = time_treshold_ms
        self._last_press_time = {'a': 0, 'w': 0, 'd': 0}

    def update(self, rec):
        self._lock.acquire()
        self._curr_rectangles = rec
        self._lock.release()
        if len(self._original_rectangles) < 3:
            self._update_original(rec)

    def _update_original(self, rec):
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
                missing_indices.append(i)
        return missing_indices

    def _can_press_key(self, key):
        current_time = time.time() * 1000  # Convert current time to milliseconds
        last_press = self._last_press_time[key]
        # Check if enough time has passed since the last key press
        if current_time - last_press >= self._time_treshold_ms:
            self._last_press_time[key] = current_time
            return True
        return False

    def start(self):
        self._stopped = False
        t = Thread(target=self._run)
        t.start()

    def _run(self):
        while not self._stopped:
            time.sleep(0.01)  # Increase Frames as this puts a huge load on CPU
            if self._curr_rectangles is not None and len(self._curr_rectangles) >= 1:
                # Get missing indices
                missing = self._find_missing_indices(self._curr_rectangles)
                # Taking Actions
                if 0 in missing and self._can_press_key('a'):
                    pyautogui.press('a')
                if 1 in missing and self._can_press_key('w'):
                    pyautogui.press('w')
                if 2 in missing and self._can_press_key('d'):
                    pyautogui.press('d')

    def stop(self):
        self._stopped = True
