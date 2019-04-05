import cv2
import torch
import pyvjoy
from mss.windows import MSS as mss
import numpy as np
from train import Alexnet
from resnet import ResNet
from nvidia_net import NvidiaNet
from conv_nets import MobileNetV2
from torchvision import transforms

import keyboard
from pynput.keyboard import Key, Controller
import time

import conv_nets
from train import get_control_str

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

    def drive(self):
        self.model.eval()
        mon = {"top": 32, "left": 0, "width": 800, "height": 600}
        sct = mss()

        self.ctlr.set_axis(axis[0], 0x4000)
        self.ctlr.set_axis(axis[1], 0x0000)
        self.ctlr.set_axis(axis[2], 0x0000)
        while True:

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

                out = self.model(img.view(1, *img.shape))[0]
                print(out)
                out = (out + 1) * 0x4000
                
                for i, ax in enumerate(axis):
                    self.ctlr.set_axis(ax, int(out[i].item()))

if __name__ == "__main__":
    # alex = conv_nets.MobileNetV2(num_classes=3)
    # alex.load_state_dict(torch.load('./models/mobilenetv2.pt'))
    alex = Alexnet()
    alex.load_state_dict(torch.load('./models/alexnet.pt'))
    juan = Driver(alex)
    juan.drive()