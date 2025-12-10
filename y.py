import wmi

c = wmi.WMI()
for gpu in c.Win32_VideoController():
    print(f"Name: {gpu.Name}, DeviceID: {gpu.DeviceID}")
