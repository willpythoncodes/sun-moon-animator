#Requiered libraries needed to run python code
#pip install tqdm
#pip install opencv-python opencv-python-headless
#Coded by William G.
#Free for fair use
#Inteded for use with Vaoinis telescopes
import os
import cv2
import numpy as np
from tkinter import filedialog, Tk
from tqdm import tqdm

def get_directory():
    """Prompt user to select directory and return its path."""
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def valid_image_check(img, ref_shape, previous_images):
    """Check if the image is valid based on shape, content, brightness distribution, and consistency with previous images."""
    
    if img is None:
        return False, "Error reading the image.", False
    
    # Check shape
    if img.shape != ref_shape:
        return False, "Shape mismatch with the reference image.", False
    
    # Check if the image is almost black (heuristic: >90% pixels are black)
    if (img < 10).all(axis=2).sum() > 0.9 * img.size // 3:
        return False, "Image is almost black.", False
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate the standard deviation of the brightness
    std_dev = np.std(gray)
    
    # A threshold for the standard deviation can be set based on testing.
    if std_dev > 70:  # This threshold value can be tweaked default is 50
        return False, f"High brightness standard deviation: {std_dev:.2f}", False
    
    if len(previous_images) == 3:
        avg_previous = np.mean(previous_images, axis=0)
        if np.abs(np.mean(gray - avg_previous)) > 15:  # 15 is a threshold which can be adjusted
            return False, "High deviation from the average of previous images.", True

    return True, "", False

def center_object(img, threshold_value=200):
    """Center the primary bright object (e.g., sun or moon) in the image using thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to filter out all but the brightest areas
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find the contours of the bright areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img  # Return the original image if no contours were found
    
    # Find the largest contour (assuming it corresponds to the sun)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute the centroid of the largest contour
    M = cv2.moments(largest_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Shift the image to center the bright object
    dX = img.shape[1] // 2 - cX
    dY = img.shape[0] // 2 - cY
    matrix = np.float32([[1, 0, dX], [0, 1, dY]])
    centered_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    
    return centered_img


def create_video_from_images(folder_path, output_name="output.mp4"):
    files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpeg')])
    if not files:
        print("No jpeg images found in the directory.")
        return
    ref_img = cv2.imread(files[0])
    if ref_img is None:
        print("Error reading the first image.")
        return
    ref_shape = ref_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(folder_path, output_name)
    out = cv2.VideoWriter(output_path, fourcc, 24.0, (ref_shape[1], ref_shape[0]))
    previous_images_buffer = []

    for file in tqdm(files, unit="image", desc="Processing"):
        try:
            img = cv2.imread(file)
            img_centered = center_object(img)
            is_valid, reason, discard_buffer = valid_image_check(img_centered, ref_shape, previous_images_buffer)
            
            if is_valid:
                out.write(img_centered)
                if len(previous_images_buffer) == 3:
                    previous_images_buffer.pop(0)
                gray = cv2.cvtColor(img_centered, cv2.COLOR_BGR2GRAY)
                previous_images_buffer.append(gray)
            else:
                print(f"Image {file} discarded. Reason: {reason}")
                if discard_buffer:
                    previous_images_buffer.clear()

        except Exception as e:
            print(f"Error processing {file}. Error: {e}")

    out.release()

if __name__ == '__main__':
    directory = get_directory()
    if directory:
        try:
            create_video_from_images(directory)
            print("Video processing completed.")
        except Exception as e:
            print(f"An error occurred during processing: {e}")
    else:
        print("No directory selected.")

