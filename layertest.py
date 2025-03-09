import cv2
import keras
from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/keras_model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/labels.txt", "r").readlines()

# Open the video file
video_path = "C:/Users/hp/Desktop/Aditya/pro files/teachable machine/PS2 - Made with Clipchamp (1).mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame extraction settings
frame_interval = 2  # Process every 5th frame
frame_count = 0

while True:
    ret, frame = cap.read()  # Read frame-by-frame
    if not ret:
        break  # Exit if no more frames

    # Process only every nth frame
    if frame_count % frame_interval == 0:
        # Convert the frame to PIL format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize and preprocess the frame
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Predict the class
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Print result for this frame
        print(f"Frame {frame_count}: Class = {class_name}, Confidence = {confidence_score:.2f}")

    frame_count += 1

# Release the video capture
cap.release()
cv2.destroyAllWindows()
