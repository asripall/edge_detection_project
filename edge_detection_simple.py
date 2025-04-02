import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the image path
image_path = '/Users/asrithapallaki/Downloads/BSR/BSDS500/data/images/test/100039.jpg'

#Check if the image exists
if not os.path.exists(image_path):
    print(f"Image not found at {image_path}")
    print("Please update the path to point to a valid image.")
    exit()

# Make sure the image loads
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Convert to grey
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges_canny = cv2.Canny(gray, 50, 150)

# Apply Sobel edge detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save image
cv2.imwrite('original.jpg', img)
cv2.imwrite('edges_canny.jpg', edges_canny)
cv2.imwrite('edges_sobel.jpg', sobel_magnitude)

print("Edge detection completed!")
print("Results saved as 'original.jpg', 'edges_canny.jpg', and 'edges_sobel.jpg'")
print("You can open these files to see the results.")

try:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(edges_canny, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_results.png')
    plt.show()
except Exception as e:
    print(f"Could not display the plot: {e}")
    print("But the result images were saved successfully.")

