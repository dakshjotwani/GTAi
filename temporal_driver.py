import cv2
import torch
import pyvjoy
import time
from mss.windows import MSS as mss
import numpy as np
from torchvision import transforms
import keyboard
from pynput.keyboard import Key, Controller

from test_lstm.alex_lstm import AlexLSTM

axis = [pyvjoy.HID_USAGE_X, pyvjoy.HID_USAGE_Y, pyvjoy.HID_USAGE_Z]

class TemporalDriver:
    def __init__(self, model, vJoyID=1):
        self.ctlr = pyvjoy.VJoyDevice(vJoyID)
        model.eval()
        self.model = model
        self.driving = False
        self.transform = transforms.ToTensor()
        self.cheatcodedelay = 0

    def cheat_code(self, code):
        if time.time() - self.cheatcodedelay < 1:
            return
        keyboard = Controller()
        for n in code:
            keyboard.press(n)
            keyboard.release(n)
        self.cheatcodedelay = time.time()
    
    def drive(self, device):
        mon = {"top": 32, "left": 0, "width": 800, "height": 600}
        sct = mss()

        hidden = None

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
                hidden = None
                print('You have arrived!')
            if keyboard.is_pressed('`'):
                self.cheat_code('RAPIDGT\n')

            if self.driving:
                img = np.asarray(sct.grab(mon))
                img = cv2.resize(img, (400, 300))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self.transform(img)
                img = img.unsqueeze(0).unsqueeze(0).to(device)
                if hidden is None:
                    out, hidden = self.model(img)
                else:
                    out, hidden = self.model(img, hidden)
                # print(out)
                out = out.squeeze(0).squeeze(0)
                print(out)
                out = (out + 1) * 0x4000
                for i, ax in enumerate(axis):
                    self.ctlr.set_axis(ax, int(out[i].item()))

                proc_time = time.time() - start
                wait_time = 1/10 - proc_time
                time.sleep(wait_time if wait_time > 0 else 0)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = AlexLSTM()
    name = 'alexlstm_3'
    model.load_state_dict(torch.load('./models/' + name + '.pt'))
    model.to(device)
    juan = TemporalDriver(model)
    juan.drive(device)