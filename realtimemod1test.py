import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Load the model
model = load_model("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/keras_model.h5", compile=False)

# Load class labels
class_names = open("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/labels.txt", "r").readlines()

# Open video file
video_path = "C:/Users/hp/Desktop/Aditya/pro files/teachable machine/video.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Set frame skip for speed (adjust as needed)
frame_skip = 2  # Process every 2nd frame

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Resize and preprocess frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32) / 127.5 - 1
    data = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Overlay class name and confidence on the video
    text = f"{class_name} ({confidence_score:.2f})"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show video with annotation
    cv2.imshow("Real-time Classification", frame)

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
