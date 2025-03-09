import os
from PIL import Image

# Input and output folders
input_folder = "C:/Users/hp/Desktop/train/angle__state_1"
output_folder = "C:/Users/hp/Desktop/to use/good_augmented_ns-1 Copy"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check image format
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Open the image
        image = Image.open(input_path)

        # Resize to (224, 224)
        image = image.resize((224, 224), Image.LANCZOS)

        

        # Crop the image (left=60, top=70, right=150, bottom=125)
        cropped_image = image.crop((40, 23,75 , 165))

        # Save the processed image
        cropped_image.save(output_path)

        print(f"Processed: {file_name}")

print("✅ Processing complete. Images saved in:", output_folder)
