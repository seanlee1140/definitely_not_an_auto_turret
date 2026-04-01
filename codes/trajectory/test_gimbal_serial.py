"""
test_gimbal_serial.py
======================

Quick sanity-check for the ESP32 serial link.
Run this BEFORE main.py to verify the firmware is responding correctly.

Usage:
    python test_gimbal_serial.py            # uses /dev/ttyUSB0
    python test_gimbal_serial.py COM3       # Windows
    python test_gimbal_serial.py /dev/ttyUSB1
"""

import sys
import time
from gimbal_controller import GimbalController

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'

print(f"Connecting to {PORT} ...")
g = GimbalController(port=PORT)

print("STATUS:", g.status())

print("Homing...")
g.home()
time.sleep(0.5)

print("Tilt to 60° (up)")
g.tilt_to(60)
time.sleep(0.8)

print("Tilt to 120° (down)")
g.tilt_to(120)
time.sleep(0.8)

print("Tilt back to 90° (level)")
g.tilt_to(90)
time.sleep(0.5)

print("Pan right 45°")
g.pan_to(45)
time.sleep(0.5)

print("Pan left 45°")
g.pan_to(-45)
time.sleep(0.5)

print("Home")
g.home()

print("STATUS:", g.status())
g.close()
print("Done — gimbal serial link OK")
