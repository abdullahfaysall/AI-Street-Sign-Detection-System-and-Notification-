import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import threading

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--notification', help='Enable/disable visual notifications (default: on)',
                    choices=['on', 'off'], default='on')
parser.add_argument('--audio', help='Enable/disable audio feedback (default: on)',
                    choices=['on', 'off'], default='on')
parser.add_argument('--reminder', help='Enable/disable reminders (default: on)',
                    choices=['on', 'off'], default='on')
parser.add_argument('--reminder-duration', help='Duration before showing reminder in seconds (default: 15)',
                    choices=['15', '30'], default='15')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Parse new control settings
show_notification = args.notification == 'on'
enable_audio = args.audio == 'on'
enable_reminder = args.reminder == 'on'
reminder_interval = int(args.reminder_duration)

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        # Get the camera's native resolution
        native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Only set resolution if it's different from native
        if resW != native_width or resH != native_height:
            ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
            # Set zoom to 1.0 (normal)
            cap.set(cv2.CAP_PROP_ZOOM, 1.0)
            # Set focus to auto
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize TTS engine only if audio is enabled
if enable_audio:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speaking rate (optional)

def speak_sign(text):
    if enable_audio:
        engine.say(text)
        engine.runAndWait()

# Initialize control and status variables
img_count = 0
notification_duration = 3  # Duration to show notification in seconds
reminder_interval = 20  # Time in seconds before showing reminder
reminder_display_duration = 5  # Duration to show reminder notification in seconds
last_notification_time = 0
current_notification = None
current_sign_image = None
reminder_notification = None
reminder_sign_image = None
reminder_scheduled = set()  # Set to track which signs have reminders scheduled
show_settings_panel = True  # Control panel visibility
reminder_start_time = 0  # Track when reminder was shown

# Define important signs that should trigger reminders
IMPORTANT_SIGNS = ['max speed 100km/h', 'caution accident area']

def schedule_reminder(sign_name, sign_image):
    if enable_reminder and sign_name in IMPORTANT_SIGNS:
        time.sleep(reminder_interval)
        if sign_name in reminder_scheduled:
            global reminder_notification, reminder_sign_image, reminder_start_time
            reminder_notification = sign_name
            reminder_sign_image = sign_image
            reminder_start_time = time.time()  # Record when reminder was shown
            reminder_scheduled.remove(sign_name)

def draw_settings_panel(frame):
    # Create a semi-transparent overlay for settings panel
    overlay = frame.copy()
    panel_width = 250  # Increased width to show controls
    panel_height = 150  # Increased height to show controls
    margin = 10
    
    # Draw semi-transparent background
    cv2.rectangle(overlay, 
                 (frame.shape[1] - panel_width - margin, margin),
                 (frame.shape[1] - margin, margin + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw settings text with controls
    settings = [
        f"Notification: {'ON' if show_notification else 'OFF'} (Press 'N')",
        f"Audio: {'ON' if enable_audio else 'OFF'} (Press 'A')",
        f"Reminder: {'ON' if enable_reminder else 'OFF'} (Press 'R')",
        f"Reminder Time: {reminder_interval}s (Press 'T')",
        f"Press 'H' to show/hide this panel"
    ]
    
    for i, setting in enumerate(settings):
        y_pos = margin + 30 + (i * 25)
        cv2.putText(frame, setting,
                   (frame.shape[1] - panel_width - margin + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Create a copy of the frame for drawing
    display_frame = frame.copy()

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.5:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(display_frame, (xmin,ymin), (xmax,ymax), color, 2)

            # Draw label with confidence
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(display_frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(display_frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Extract the detected sign region
            sign_region = frame[ymin:ymax, xmin:xmax]
            
            # Resize the sign region to a smaller size for display
            sign_height = 50  # Reduced height for notification
            aspect_ratio = sign_region.shape[1] / sign_region.shape[0]
            sign_width = int(sign_height * aspect_ratio)
            small_sign = cv2.resize(sign_region, (sign_width, sign_height))

            # Update notification and sign image
            current_time = time.time()
            if current_time - last_notification_time > notification_duration:
                current_notification = classname
                current_sign_image = small_sign
                last_notification_time = current_time
                
                # Speak the sign name in a separate thread if audio is enabled
                if enable_audio:
                    threading.Thread(target=speak_sign, args=(classname,), daemon=True).start()
                
                # Schedule reminder if enabled and not already scheduled and is an important sign
                if enable_reminder and classname in IMPORTANT_SIGNS and classname not in reminder_scheduled:
                    reminder_scheduled.add(classname)
                    threading.Thread(target=schedule_reminder, args=(classname, small_sign.copy()), daemon=True).start()

    # Display notification if active and enabled
    if show_notification and current_notification and time.time() - last_notification_time < notification_duration:
        # Create semi-transparent overlay for notification
        overlay = display_frame.copy()
        notification_height = 70
        cv2.rectangle(overlay, (10, display_frame.shape[0]-notification_height), 
                     (display_frame.shape[1]-10, display_frame.shape[0]-10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Add sign image to notification
        if current_sign_image is not None:
            img_y = display_frame.shape[0] - notification_height + 10
            img_x = 20
            display_frame[img_y:img_y+current_sign_image.shape[0], 
                         img_x:img_x+current_sign_image.shape[1]] = current_sign_image
        
        # Add notification text next to the sign image
        text_x = 20 + current_sign_image.shape[1] + 10 if current_sign_image is not None else 20
        text_y = display_frame.shape[0] - notification_height + 35
        cv2.putText(display_frame, current_notification, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display reminder notification if active and enabled
    if show_notification and enable_reminder and reminder_notification is not None:
        current_time = time.time()
        # Only show reminder for reminder_display_duration seconds
        if current_time - reminder_start_time < reminder_display_duration:
            # Create semi-transparent overlay for reminder notification
            overlay = display_frame.copy()
            reminder_height = 70
            cv2.rectangle(overlay, (10, 10), 
                         (display_frame.shape[1]-10, reminder_height+10), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
            
            # Add reminder sign image
            if reminder_sign_image is not None:
                img_y = 20
                img_x = 20
                display_frame[img_y:img_y+reminder_sign_image.shape[0], 
                             img_x:img_x+reminder_sign_image.shape[1]] = reminder_sign_image
            
            # Add reminder text
            text_x = 20 + reminder_sign_image.shape[1] + 10 if reminder_sign_image is not None else 20
            text_y = reminder_height - 25
            cv2.putText(display_frame, f"Reminder: {reminder_notification}", 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Clear reminder after duration expires
            reminder_notification = None
            reminder_sign_image = None

    # Draw settings panel if enabled
    if show_settings_panel:
        draw_settings_panel(display_frame)

    # Display detection results
    cv2.namedWindow('YOLO detection results', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('YOLO detection results', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('YOLO detection results', display_frame)
    if record: recorder.write(display_frame)

    # Handle keyboard input
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png', display_frame)
    elif key == ord('n') or key == ord('N'): # Toggle notifications
        show_notification = not show_notification
    elif key == ord('a') or key == ord('A'): # Toggle audio
        enable_audio = not enable_audio
        if enable_audio and 'engine' not in locals():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
    elif key == ord('r') or key == ord('R'): # Toggle reminders
        enable_reminder = not enable_reminder
    elif key == ord('t') or key == ord('T'): # Toggle reminder duration
        reminder_interval = 30 if reminder_interval == 15 else 15
    elif key == ord('h') or key == ord('H'): # Toggle settings panel visibility
        show_settings_panel = not show_settings_panel

# Clean up
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
