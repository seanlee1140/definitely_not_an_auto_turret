# Check if the camera is detected
ls /dev/video*

# Quick test capture with GStreamer
# For CAM0 (sensor-id=0)
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! videoconvert ! xvimagesink

# For CAM1 (sensor-id=1)
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! videoconvert ! xvimagesink