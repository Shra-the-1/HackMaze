import cv2
import os

# Input and output directories
input_folder = "C:/Users/hp/Downloads/DEFECTS-20250309T050219Z-001/DEFECTS/angle_1"   # Folder containing images
output_folder = "C:/Users/hp/Downloads/DEFECTS-20250309T050219Z-001/DEFECTS/output_images_defect/angle_1_defects" # Folder to save cropped images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define cropping region (x_start, y_start, width, height)
crop_x, crop_y, crop_w, crop_h = 470, 220 , 350 , 205  # Adjust as needed

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Skipping {filename}, unable to read.")
            continue

        # Crop the image
        cropped_image = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        # Save the cropped image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped and saved: {output_path}")

print("Batch cropping complete!")
