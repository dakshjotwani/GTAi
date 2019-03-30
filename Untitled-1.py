import time

import cv2
from mss.windows import MSS as mss
import pygame
import numpy as np
import datetime
import keyboard
from pynput.keyboard import Key, Controller

font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

class FrameInputCapture:
    def __init__(self, fraps=25):
        self.fraps = fraps
        self.frames = []
        self.recording = False

    def put_turn_text(self, img, turn):
        turn_text = 'Straight'
        if turn > 0.01:
            turn_text = 'Right: ' + str(turn)
        elif turn < -0.01:
            turn_text = 'Left: ' + str(-turn)

        cv2.putText(img, turn_text,
            (10, 500),
            font,
            fontScale,
            fontColor,
            lineType)

    def put_gas_text(self, img, gas):
        cv2.putText(img, 'Gas: ' + str((gas + 1) / 2),
            (10, 550),
            font,
            fontScale,
            fontColor,
            lineType)

    def put_brake_text(self, img, brake):
        cv2.putText(img, 'Brake: ' + str((brake + 1) / 2),
            (10, 600),
            font,
            fontScale,
            fontColor,
            lineType)

    def save_to_buffer(self, img, controls):
        if not self.recording:
            return
        self.frames.append((cv2.resize(img, (400,300)), controls))

    def cheat_code(self, code):
        keyboard = Controller()
        keyboard.press('~')
        keyboard.release('~')
        for n in code:
            keyboard.press(n)
            keyboard.release(n)
        keyboard.press('\n')
        keyboard.release('\n')

    
    def flush_buffer(self):
        cur_time = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
        control_labels = ''
        for i, n in enumerate(self.frames[:-self.fraps*2]): #remove last two seconds
            filename = cur_time + '--' + str(i) + '.jpg'
            cv2.imwrite('./gta5train/' + filename, n[0], [cv2.IMWRITE_JPEG_QUALITY, 60])
            control_labels += filename + '\t' + '\t'.join(map(str, n[1])) + '\n'
        
        with open('./gta5train/' + cur_time + '.txt', 'w') as f:
            f.write(control_labels)

        self.frames = []
        print("Flushed", cur_time)

    def screen_record(self, debug=False):
        # pygame init bs
        pygame.init()
        # Initialize controller
        ctlr = pygame.joystick.Joystick(1)
        ctlr.init()

        # mss init bs
        mon = {"top": 32, "left": 0, "width": 800, "height": 600}
        title = "Screen Capture"
        sct = mss()

        # Keep track of inputs
        turn = 0.0
        gas = -1.0
        brake = -1.0

        while True:
            start = time.time()
            # Get frame
            img = np.asarray(sct.grab(mon))

            if keyboard.is_pressed('q') and not self.recording:
                exit()
            if keyboard.is_pressed('i') and not self.recording:
                self.recording = True
                print('Started recording...')
            if keyboard.is_pressed('o') and self.recording:
                self.recording = False
                self.flush_buffer()
            if keyboard.is_pressed('~'):
                self.cheat_code('RAPIDGT')

            pygame.event.pump()
            turn = ctlr.get_axis(0)
            gas = ctlr.get_axis(4)
            brake = ctlr.get_axis(5)

            self.save_to_buffer(img, np.array([turn, gas, brake], dtype=np.float16))

            # cv2 bs
            self.put_turn_text(img, turn)
            self.put_gas_text(img, gas)
            self.put_brake_text(img, brake)

            cv2.imshow(title, img)

            proc_time = time.time() - start
            wait_time = int(1000/(self.fraps) - proc_time)
            if cv2.waitKey(wait_time if wait_time > 0 else 1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
            #print(1/(time.time() - start))

        sct.close()

if __name__ == "__main__":
    fic = FrameInputCapture()
    fic.screen_record(debug=True)