
import os
from PIL import Image, ImageOps

# Input and output folders
input_folder = 'C:/Users/hp/Desktop/Aditya/pro files/teachable machine/cropped'
output_folder = "C:/Users/hp/Desktop/to use/good_s"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check image format
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Open the image
        image = Image.open(input_path)

        # Get original dimensions
        width, height = image.size
        max_dim = max(width, height)  # Find the larger dimension

        # Create a new square image with a white background
        square_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))

        # Paste the original image at the center
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        square_image.paste(image, (paste_x, paste_y))

        # Save the processed image
        square_image.save(output_path)

        print(f"Processed: {file_name}")

print("✅ All images are now squared and saved in:", output_folder)
