import time
import cv2
import torch
import pyvjoy
import numpy as np
import keyboard

from torchvision import transforms
from mss.windows import MSS as mss
from pynput.keyboard import Key, Controller

from models import GTAResNet

axis = [pyvjoy.HID_USAGE_X, pyvjoy.HID_USAGE_Y, pyvjoy.HID_USAGE_Z]

class Driver:
    def __init__(self, model, vJoyID=1):
        self.ctlr = pyvjoy.VJoyDevice(1)
        self.model = model
        self.driving = False
        self.transform = transforms.ToTensor()
        self.cheatcodedelay = 0

    
    def cheat_code(self, code):
        if time.time() - self.cheatcodedelay < 1:
            return
        time.sleep(0.5)
        keyboard = Controller()
        for n in code:
            keyboard.press(n)
            keyboard.release(n)
        self.cheatcodedelay = time.time()

    def drive(self, device):
        self.model.eval()
        mon = {"top": 32, "left": 0, "width": 800, "height": 600}
        sct = mss()

        self.ctlr.set_axis(axis[0], 0x4000)
        self.ctlr.set_axis(axis[1], 0x0000)
        self.ctlr.set_axis(axis[2], 0x0000)
        while True:
            start = time.time()

            if keyboard.is_pressed('q'):
                exit()
            if keyboard.is_pressed('i') and not self.driving:
                self.driving = True
                print('Started driving...')
            if keyboard.is_pressed('o') and self.driving:
                self.driving = False
                self.ctlr.set_axis(axis[0], 0x4000)
                self.ctlr.set_axis(axis[1], 0x0000)
                self.ctlr.set_axis(axis[2], 0x0000)
                print('You have arrived!')
            if keyboard.is_pressed('`'):
                self.cheat_code('RAPIDGT\n')

            if self.driving:
                img = np.asarray(sct.grab(mon))
                img = cv2.resize(img, (400, 300))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)

                out = self.model(img.view(1, *img.shape).to(device))[0]
                print(out)
                out = (out + 1) * 0x4000
                
                for i, ax in enumerate(axis):
                    self.ctlr.set_axis(ax, int(out[i].item()))
                
                #print(1 / (time.time() - start))

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    alex = GTAResNet()
    alex.load_state_dict(torch.load('../models/FinalResNet50-2.pt'))
    alex.to(device)
    juan = Driver(alex)
    juan.drive(device)