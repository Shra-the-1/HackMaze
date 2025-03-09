import cv2
import keras
from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/keras_model.h5", compile=False)

model2 = load_model("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/keras_model2.h5", compile=False)

# Load the labels
class_names = open("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/labels.txt", "r").readlines()
class_names2 = open("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/labels2.txt", "r").readlines()

# Open the video file
video_path = "C:/Users/hp/Desktop/Aditya/pro files/teachable machine/PS2 - Made with Clipchamp (1).mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame extraction settings
frame_interval = 2  # Process every 2th frame
frame_count = 0

outputpath = "C:/Users/hp/Desktop/Aditya/pro files/teachable machine/cropped"
os.makedirs(outputpath,exist_ok=True)


def make_square(image, background_color=(255, 255, 255)):
    """
    Converts a PIL Image into a square by adding padding.

    :param image: PIL Image object.
    :param background_color: Background color for padding (default: white).
    :return: PIL Image object (squared image).
    """
    width, height = image.size
    new_size = max(width, height)  # Determine square size

    # Create a new square canvas
    square_img = Image.new("RGB", (new_size, new_size), background_color)

    # Calculate position to center the original image
    left = (new_size - width) // 2
    top = (new_size - height) // 2
    square_img.paste(image, (left, top))

    return square_img  # Return squared image


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
        #image = np.array(image)  # Convert to NumPy array
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #cv2.imshow("abcd", image)
        #cv2.waitKey(100)
        #image.save(os.path.join(outputpath, f"frame_{frame_count}.jpg"))

        #input pasth,output path change, predicted_class =model.predict()

        predicted_class=index
        pil_img = image
        #pil_image = pil_img.resize(224,224)
        width, height = pil_img.size  # Assume images are 224x224

        if predicted_class == 4:
            #new_width, new_height = (width * 5) // 26, (height * 4) // 15
            """left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height"""
            cropped_image = image.crop((60,70,150,125))

        elif predicted_class == 2:
            """new_width, new_height = (width * 9.5) // 26, (height * 5) // 15
            left = (width - new_width) // 2
            right = left + new_width
            top = (4.5 * height) // 15
            bottom = top + new_height"""
            cropped_image = image.crop((60,90,190,160))

        elif predicted_class == 0:
            '''new_width, new_height = (width * 3) // 26, (height * 10.5) // 15
            left = (5.5 * width) // 26
            right = left + new_width
            top = (3 * height) // 15
            bottom = top + new_height'''
            cropped_image = image.crop((40, 23,75 , 165))

        else:
            continue  # Skip if class is not handled

        # Crop the image
        #cropped_img = pil_img.crop((left, top, right, bottom))
        #cropped_img = np.array(cropped_img)  # Convert to NumPy array
        #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        print("done")
        
        cropped_image.save(os.path.join(outputpath, f"frame_{frame_count}.jpg"))
        #cv2.imshow("abcd", cropped_img)
        #cv2.waitKey(1)'''
        

        
        squared_image=make_square(cropped_image)
        cropped_image_resized = ImageOps.fit(squared_image, (224, 224), Image.Resampling.LANCZOS)
        cropped_array = np.asarray(cropped_image_resized)
        normalized_cropped_array = (cropped_array.astype(np.float32) / 127.5) - 1
        data2 = np.expand_dims(normalized_cropped_array, axis=0)

        prediction2 = model2.predict(data2)
        index2 = int(np.argmax(prediction2))
        class_name2 = class_names2[index2]
        confidence2 = prediction2[0][index2]
        print(f"Frame {frame_count}: Model2 predicted: {class_name2}, Confidence: {confidence2:.2f}")


    frame_count += 1

# Release the video capture
cap.release()
cv2.destroyAllWindows()
