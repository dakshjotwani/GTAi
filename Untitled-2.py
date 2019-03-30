import pyvjoy
import time

j = pyvjoy.VJoyDevice(1)

#turn button number 15 on
#j.set_button(8,1)
##time.sleep(5)
#Notice the args are (buttonID,state) whereas vJoy's native API is the other way around.

#turn button 15 off again
#j.set_button(8,0)
#time.sleep(5)

print(input())

#Set X axis to fully left
j.set_axis(pyvjoy.HID_USAGE_X, 0x1)
time.sleep(1)

#Set X axis to fully right
j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)
time.sleep(1)

#Reset
j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
time.sleep(1)

j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
time.sleep(1)

j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
time.sleep(1)

j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
time.sleep(1)

j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
time.sleep(1)

j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
time.sleep(1)

j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
time.sleep(1)

j.reset()