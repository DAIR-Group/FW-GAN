import os
import argparse
import cv2
import numpy as np
from PIL import Image

def process_style_image(input_path, output_path, height=32):
    """
    Processes a handwriting image for FW-GAN:
    1. Grayscale
    2. Adaptive Thresholding to clean background
    3. Resize to 32px height
    4. Save as PNG
    """
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to remove shadows and make background white
    # Block size and C might need tuning based on image quality
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Optional: Denoising
    binary = cv2.medianBlur(binary, 3)

    # Resize to target height while maintaining aspect ratio
    h, w = binary.shape
    new_w = int(w * (height / h))
    resized = cv2.resize(binary, (new_w, height), interpolation=cv2.INTER_AREA)

    # Save processed image
    cv2.imwrite(output_path, resized)
    print(f"Processed style image saved to: {output_path}")
    print(f"Original size: {w}x{h}, New size: {new_w}x{height}")

def main():
    parser = argparse.ArgumentParser(description="Process handwriting image for FW-GAN style encoding.")
    parser.add_argument("--input", type=str, required=True, help="Path to your handwriting image")
    parser.add_argument("--output", type=str, default="processed_style.png", help="Path to save the processed image")
    parser.add_argument("--height", type=int, default=32, help="Target height for the image (default 32)")
    
    args = parser.parse_args()
    
    process_style_image(args.input, args.output, args.height)

if __name__ == "__main__":
    main()
