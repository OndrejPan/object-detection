import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO


class ObjectDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Object Detection with YOLOv5")

        # Change window background color to light blue
        self.window.configure(background='#ADD8E6')

        # Load the YOLO model
        self.model = YOLO(r"C:\Users\panik\PycharmProjects\pythonProject2\runs\detect\train3\weights\yolov8_custom.pt")

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        # Create start, stop, and close buttons
        self.start_button = tk.Button(window, text="Zapni", width=10, command=self.start, bg='#0066cc', fg='#ffffff',
                                      font=('Arial', 12, 'bold'), borderwidth=0, border=0, activebackground='#0052cc',
                                      activeforeground='#ffffff')
        self.start_button.grid(row=1, column=0, padx=10, pady=10)

        self.stop_button = tk.Button(window, text="Zastav", width=10, command=self.stop, bg='#0066cc', fg='#ffffff',
                                     font=('Arial', 12, 'bold'), borderwidth=0, border=0, activebackground='#0052cc',
                                     activeforeground='#ffffff')
        self.stop_button.grid(row=1, column=1, padx=10, pady=10)

        self.close_button = tk.Button(window, text="Vypni", width=10, command=self.close, bg='#0066cc', fg='#ffffff',
                                      font=('Arial', 12, 'bold'), borderwidth=0, border=0, activebackground='#0052cc',
                                      activeforeground='#ffffff')
        self.close_button.grid(row=1, column=2, padx=10, pady=10)

        # Create a Combobox for selecting masking method
        self.mask_method_label = tk.Label(window, text="Masking Method:", bg='#ADD8E6', font=('Arial', 12))
        self.mask_method_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        self.mask_methods = ttk.Combobox(window, values=["Ziadne", "HSV Masking", "Threshold Masking"], width=20)
        self.mask_methods.current(0)  # Set default value to "None"
        self.mask_methods.grid(row=0, column=1, padx=10, pady=10)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.grid(row=2, columnspan=3)

        self.is_running = False

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.show_video()

    def stop(self):
        self.is_running = False

    def close(self):
        self.is_running = False
        self.cap.release()
        self.window.quit()

    def show_video(self):
        ret, frame = self.cap.read()

        if ret:
            # Select the masking method based on Combobox selection
            method = self.mask_methods.get()
            if method == "HSV Masking":
                frame = self.hsv_mask(frame)
            elif method == "Threshold Masking":
                frame = self.threshold_mask(frame)

            # Make a prediction on the frame
            results = self.model(frame)

            # Get the bounding boxes and class labels for the detected objects
            boxes = results[0].boxes.xyxy.numpy()
            labels = results[0].boxes.cls.numpy().astype(int)

            # Process detected objects and track centroids
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)

                # Only consider detections for your target class (adjust class ID)
                if label == 0:  # Assuming class 0 is your target object

                    # Draw bounding box and label (modify text if needed)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{self.model.names[label]}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

            # Convert the frame to RGB format and display in GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)

            self.canvas.img = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

        if self.is_running:
            self.window.after(10, self.show_video)

    def hsv_mask(self, frame):
        # Apply HSV masking to the frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = (40, 40, 40)
        upper_green = (70, 255, 255)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

    def threshold_mask(self, frame):
        # Apply threshold masking to the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame


# Create a Tkinter window
window = tk.Tk()
app = ObjectDetectionApp(window)
window.mainloop()
