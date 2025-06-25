import sys
import cv2
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QCheckBox, 
                            QComboBox, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import threading
from ultralytics import YOLO
import pyttsx3

# Define important signs that should trigger reminders
IMPORTANT_SIGNS = ['max speed 100km/h', 'caution accident area']

def speak_sign(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

class TrafficSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Sign Detection")
        
        # Get screen size and set window size
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        # Remove window frame and make it borderless
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Initialize variables
        self.model = None
        self.cap = None
        self.current_notification = None
        self.current_sign_image = None
        self.reminder_notification = None
        self.reminder_sign_image = None
        self.show_notification = True
        self.enable_audio = True
        self.enable_reminder = True
        self.reminder_interval = 15
        self.reminder_scheduled = set()
        self.last_notification_time = 0
        self.notification_duration = 3
        self.reminder_display_duration = 5
        self.reminder_start_time = 0
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create left panel for camera and notifications
        left_panel = QWidget()
        left_panel.setMinimumWidth(int(screen.width() * 0.75))
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(int(screen.width() * 0.75), int(screen.height() * 0.8))
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        left_layout.addWidget(self.camera_label)

        # Create overlay for notifications and reminders
        self.overlay = QWidget(self)
        self.overlay.setGeometry(0, 0, screen.width(), screen.height())
        self.overlay.setStyleSheet("background-color: transparent;")
        self.overlay.hide()

        # Notification panel
        self.notification_panel = QFrame(self.overlay)
        self.notification_panel.setFixedSize(400, 120)
        self.notification_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(231, 76, 60, 0.5);
                border-radius: 15px;
                border: 2px solid rgba(192, 57, 43, 0.8);
            }
        """)
        notification_layout = QHBoxLayout(self.notification_panel)
        notification_layout.setContentsMargins(10, 10, 10, 10)
        notification_layout.setSpacing(10)
        
        # Add sign image label for notification
        self.notification_image = QLabel()
        self.notification_image.setFixedSize(80, 80)
        self.notification_image.setStyleSheet("""
            QLabel {
                background-color: transparent;
                padding: 5px;
                border: none;
            }
        """)
        notification_layout.addWidget(self.notification_image)
        
        self.notification_label = QLabel()
        self.notification_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: transparent;
                border: none;
            }
        """)
        self.notification_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notification_layout.addWidget(self.notification_label)
        self.notification_panel.hide()

        # Reminder panel
        self.reminder_panel = QFrame(self.overlay)
        self.reminder_panel.setFixedSize(500, 120)  # Increased width to fit text
        self.reminder_panel.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 15px;
                border: 2px solid #34495e;
            }
        """)
        reminder_layout = QHBoxLayout(self.reminder_panel)
        reminder_layout.setContentsMargins(15, 15, 15, 15)  # Increased margins
        reminder_layout.setSpacing(15)  # Increased spacing
        
        # Add sign image label for reminder
        self.reminder_image = QLabel()
        self.reminder_image.setFixedSize(80, 80)
        self.reminder_image.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        reminder_layout.addWidget(self.reminder_image)
        
        self.reminder_label = QLabel()
        self.reminder_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                min-width: 350px;  /* Ensure minimum width for text */
            }
        """)
        self.reminder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # Left align text
        self.reminder_label.setWordWrap(True)  # Enable word wrapping
        reminder_layout.addWidget(self.reminder_label)
        self.reminder_panel.hide()

        # Create right panel for settings
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f6fa;
                border-left: 1px solid #dcdde1;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)

        # Add escape button to close the application
        escape_button = QPushButton("Exit")
        escape_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        escape_button.clicked.connect(self.close)
        right_layout.addWidget(escape_button)

        # Settings title with divider
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 10, 0, 10)
        
        settings_title = QLabel("Sign Reader Settings")
        settings_title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px 0;
            }
        """)
        settings_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(settings_title)
        
        # Add a divider line
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #bdc3c7;")
        divider.setFixedHeight(2)
        title_layout.addWidget(divider)
        
        right_layout.addWidget(title_container)

        # Create a container for notification settings
        notification_group = QWidget()
        notification_layout = QVBoxLayout(notification_group)
        notification_layout.setContentsMargins(0, 0, 0, 0)
        notification_layout.setSpacing(10)

        # Notification settings title
        notification_title = QLabel("Notification Settings")
        notification_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px 0;
            }
        """)
        notification_layout.addWidget(notification_title)

        # Notification toggle
        self.notification_check = QCheckBox("Enable Notifications")
        self.notification_check.setChecked(True)
        self.notification_check.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                padding: 5px;
                color: #2c3e50;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.notification_check.stateChanged.connect(self.toggle_notifications)
        notification_layout.addWidget(self.notification_check)

        # Audio toggle
        self.audio_check = QCheckBox("Enable Audio")
        self.audio_check.setChecked(True)
        self.audio_check.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                padding: 5px;
                color: #2c3e50;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.audio_check.stateChanged.connect(self.toggle_audio)
        notification_layout.addWidget(self.audio_check)

        right_layout.addWidget(notification_group)

        # Create a container for reminder settings
        reminder_group = QWidget()
        reminder_layout = QVBoxLayout(reminder_group)
        reminder_layout.setContentsMargins(0, 0, 0, 0)
        reminder_layout.setSpacing(10)

        # Reminder settings title
        reminder_title = QLabel("Reminder Settings")
        reminder_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px 0;
            }
        """)
        reminder_layout.addWidget(reminder_title)

        # Reminder toggle
        self.reminder_check = QCheckBox("Enable Reminders")
        self.reminder_check.setChecked(True)
        self.reminder_check.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                padding: 5px;
                color: #2c3e50;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.reminder_check.stateChanged.connect(self.toggle_reminders)
        reminder_layout.addWidget(self.reminder_check)

        # Reminder duration
        duration_label = QLabel("Reminder Duration:")
        duration_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding: 5px;
                color: #2c3e50;
            }
        """)
        reminder_layout.addWidget(duration_label)
        
        # Create button container for duration selection
        duration_button_container = QWidget()
        duration_button_layout = QHBoxLayout(duration_button_container)
        duration_button_layout.setContentsMargins(0, 0, 0, 0)
        duration_button_layout.setSpacing(10)
        
        # Create duration buttons
        self.duration_15_button = QPushButton("15s")
        self.duration_30_button = QPushButton("30s")
        
        # Set initial state
        self.duration_15_button.setProperty("selected", True)
        self.duration_15_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                border: 2px solid #3498db;
                border-radius: 5px;
                color: white;
                background-color: #3498db;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #2980b9;
                border-color: #2980b9;
            }
        """)
        
        self.duration_30_button.setProperty("selected", False)
        self.duration_30_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                color: #2c3e50;
                background-color: white;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #f5f6fa;
                border-color: #3498db;
            }
        """)
        
        # Connect button clicks
        self.duration_15_button.clicked.connect(lambda: self.change_reminder_duration("15 seconds"))
        self.duration_30_button.clicked.connect(lambda: self.change_reminder_duration("30 seconds"))
        
        duration_button_layout.addWidget(self.duration_15_button)
        duration_button_layout.addWidget(self.duration_30_button)
        reminder_layout.addWidget(duration_button_container)

        right_layout.addWidget(reminder_group)

        # Add stretch to push everything to the top
        right_layout.addStretch()

        # Add panels to main layout
        layout.addWidget(left_panel, stretch=7)
        layout.addWidget(right_panel, stretch=3)

        # Initialize camera and model
        self.init_camera()
        self.init_model()

        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            sys.exit()
        print("Camera initialized on device 0")

    def init_model(self):
        try:
            self.model = YOLO("my_model.pt", task='detect')
            print("Model loaded successfully: my_model.pt")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Run detection
            results = self.model(frame, verbose=False)
            detections = results[0].boxes

            # Process detections
            for detection in detections:
                # Get bounding box coordinates
                xyxy = detection.xyxy.cpu().numpy().squeeze()
                xmin, ymin, xmax, ymax = map(int, xyxy)

                # Get class and confidence
                class_idx = int(detection.cls.item())
                class_name = self.model.names[class_idx]
                conf = detection.conf.item()

                if conf > 0.5:
                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f'{class_name}: {int(conf*100)}%'
                    cv2.putText(frame, label, (xmin, ymin-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Handle notifications and reminders
                    self.handle_detection(class_name, frame[ymin:ymax, xmin:xmax])

            # Convert frame to QImage and display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale the image to fill the label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.camera_label.width(),
                self.camera_label.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,  # Changed to fill the space
                Qt.TransformationMode.SmoothTransformation
            )
            self.camera_label.setPixmap(scaled_pixmap)

    def handle_detection(self, class_name, sign_image):
        current_time = time.time()
        
        # Print detection information
        print(f"Detected: {class_name} at (x: {int(sign_image.shape[1]/2)}, y: {int(sign_image.shape[0]/2)})")
        
        # Handle audio independently of notifications
        if self.enable_audio and current_time - self.last_notification_time > self.notification_duration:
            threading.Thread(target=speak_sign, args=(class_name,), daemon=True).start()
        
        # Handle notification
        if self.show_notification and current_time - self.last_notification_time > self.notification_duration:
            try:
                # Convert BGR to RGB for proper display
                sign_image_rgb = cv2.cvtColor(sign_image, cv2.COLOR_BGR2RGB)
                self.current_notification = class_name
                self.current_sign_image = sign_image_rgb
                self.last_notification_time = current_time
                self.show_notification_panel()
                print(f"Notification shown for: {class_name}")
                
                # Handle reminder for important signs
                if self.enable_reminder and class_name in IMPORTANT_SIGNS:
                    if class_name not in self.reminder_scheduled:
                        self.reminder_scheduled.add(class_name)
                        print(f"Reminder set for: {self.reminder_interval} seconds")
                        threading.Thread(target=self.schedule_reminder, 
                                       args=(class_name, sign_image_rgb.copy()), daemon=True).start()
            except Exception as e:
                print(f"Error handling detection: {e}")

    def show_notification_panel(self):
        try:
            if self.current_notification and self.current_sign_image is not None and self.show_notification:
                self.notification_label.setText(self.current_notification)
                
                # Convert sign image to QPixmap and display
                h, w, ch = self.current_sign_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(self.current_sign_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    80, 80,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.notification_image.setPixmap(pixmap)
                
                self.overlay.show()
                self.notification_panel.show()
                
                # Position at bottom of screen
                x = (self.width() - self.notification_panel.width()) // 2
                y = self.height() - self.notification_panel.height() - 20  # 20px from bottom
                self.notification_panel.move(x, y)
                
                # Hide after duration
                QTimer.singleShot(self.notification_duration * 1000, self.hide_notification)
        except Exception as e:
            print(f"Error showing notification: {e}")

    def hide_notification(self):
        self.notification_panel.hide()
        if not self.reminder_panel.isVisible():
            self.overlay.hide()

    def schedule_reminder(self, sign_name, sign_image):
        try:
            time.sleep(self.reminder_interval)
            if sign_name in self.reminder_scheduled and self.enable_reminder:  # Check if reminders are still enabled
                self.reminder_notification = sign_name
                self.reminder_sign_image = sign_image
                self.reminder_start_time = time.time()
                self.reminder_scheduled.remove(sign_name)
                # Use QTimer to ensure GUI updates happen in the main thread
                QTimer.singleShot(0, self.show_reminder_panel)
        except Exception as e:
            print(f"Error scheduling reminder: {e}")

    def show_reminder_panel(self):
        try:
            if self.reminder_notification and self.reminder_sign_image is not None and self.enable_reminder:
                self.reminder_label.setText(f"Reminder: {self.reminder_notification}")
                
                # Convert sign image to QPixmap and display
                h, w, ch = self.reminder_sign_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(self.reminder_sign_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    80, 80,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.reminder_image.setPixmap(pixmap)
                
                self.overlay.show()
                self.reminder_panel.show()
                
                # Position at bottom of screen
                x = (self.width() - self.reminder_panel.width()) // 2
                y = self.height() - self.reminder_panel.height() - 20  # 20px from bottom
                self.reminder_panel.move(x, y)
                
                # Hide after duration
                QTimer.singleShot(self.reminder_display_duration * 1000, self.hide_reminder)
        except Exception as e:
            print(f"Error showing reminder: {e}")

    def hide_reminder(self):
        self.reminder_panel.hide()
        if not self.notification_panel.isVisible():
            self.overlay.hide()
        self.reminder_notification = None
        self.reminder_sign_image = None

    def toggle_notifications(self, state):
        self.show_notification = state == Qt.CheckState.Checked.value
        if not self.show_notification:
            self.notification_panel.hide()
            if not self.reminder_panel.isVisible():
                self.overlay.hide()

    def toggle_audio(self, state):
        self.enable_audio = state == Qt.CheckState.Checked.value
        if not self.enable_audio:
            try:
                engine = pyttsx3.init()
                engine.stop()
            except:
                pass

    def toggle_reminders(self, state):
        self.enable_reminder = state == Qt.CheckState.Checked.value
        if not self.enable_reminder:
            self.reminder_panel.hide()
            self.reminder_scheduled.clear()
            if not self.notification_panel.isVisible():
                self.overlay.hide()

    def change_reminder_duration(self, duration):
        try:
            self.reminder_interval = int(duration.split()[0])
            # Update button styles
            if duration == "15 seconds":
                self.duration_15_button.setProperty("selected", True)
                self.duration_30_button.setProperty("selected", False)
                self.duration_15_button.setStyleSheet("""
                    QPushButton {
                        font-size: 14px;
                        padding: 8px;
                        border: 2px solid #3498db;
                        border-radius: 5px;
                        color: white;
                        background-color: #3498db;
                        min-width: 60px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                        border-color: #2980b9;
                    }
                """)
                self.duration_30_button.setStyleSheet("""
                    QPushButton {
                        font-size: 14px;
                        padding: 8px;
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                        color: #2c3e50;
                        background-color: white;
                        min-width: 60px;
                    }
                    QPushButton:hover {
                        background-color: #f5f6fa;
                        border-color: #3498db;
                    }
                """)
            else:
                self.duration_15_button.setProperty("selected", False)
                self.duration_30_button.setProperty("selected", True)
                self.duration_15_button.setStyleSheet("""
                    QPushButton {
                        font-size: 14px;
                        padding: 8px;
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                        color: #2c3e50;
                        background-color: white;
                        min-width: 60px;
                    }
                    QPushButton:hover {
                        background-color: #f5f6fa;
                        border-color: #3498db;
                    }
                """)
                self.duration_30_button.setStyleSheet("""
                    QPushButton {
                        font-size: 14px;
                        padding: 8px;
                        border: 2px solid #3498db;
                        border-radius: 5px;
                        color: white;
                        background-color: #3498db;
                        min-width: 60px;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                        border-color: #2980b9;
                    }
                """)
            
            # Clear any existing reminders when duration changes
            self.reminder_scheduled.clear()
            if self.reminder_panel.isVisible():
                self.reminder_panel.hide()
                if not self.notification_panel.isVisible():
                    self.overlay.hide()
        except ValueError:
            print("Invalid reminder duration value")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignApp()
    window.show()
    sys.exit(app.exec()) 