import cv2
import keras
from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
import os


def images_to_video(image_folder, output_video, fps=30):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No images found in the folder.")
        return
    
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f'Video saved as {output_video}')

# Example usage
images_to_video('C:/Users/hp/Downloads/to_video', 'output_video.mp4', fps=30)







# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/keras_model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/hp/Desktop/Aditya/pro files/teachable machine/labels.txt", "r").readlines()

# Open the video file
video_path = "C:/Users/hp/Desktop/Aditya/pro files/output_video.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Frame extraction settings
frame_interval = 1  # Process every 2th frame
frame_count = 0

outputpath = "C:/Users/hp/Desktop/to use/defects_ns"
os.makedirs(outputpath,exist_ok=True)


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
            cropped_image = image.crop((0,30,38,177))

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



    frame_count += 1

# Release the video capture
cap.release()
cv2.destroyAllWindows()
