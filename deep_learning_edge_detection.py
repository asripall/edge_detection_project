import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define a simple U-Net model
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Encoding
        x1 = self.encoder(x)
        
        # Middle
        x2 = self.middle(x1)
        
        # Decoding
        x3 = self.decoder(x2)
        
        return torch.sigmoid(x3)  # Sigmoid to get values between 0 and 1

# Create the model
model = SimpleUNet()

# Same image path as before
image_path = '/Users/asrithapallaki/Downloads/BSR/BSDS500/data/images/test/100039.jpg' 

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Image not found at {image_path}")
    exit()

# Load and preprocess the image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize for the model (U-Net typically expects specific dimensions)
img_resized = cv2.resize(img_rgb, (256, 256))

# Convert to PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])
img_tensor = transform(img_resized).unsqueeze(0)  # Add batch dimension

# We would normally train the model here, but for this example
# we'll just use it untrained to demonstrate the concept

# Set model to evaluation mode
model.eval()

# Process the image with the model
with torch.no_grad():
    output = model(img_tensor)

# Convert output tensor to numpy array
edge_map = output.squeeze().numpy()

# Compare results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edge_map, cmap='gray')
plt.title('U-Net (Pre-trained)')
plt.axis('off')

# Load Canny results for comparison
canny_edges = cv2.imread('edges_canny.jpg', cv2.IMREAD_GRAYSCALE)

plt.subplot(1, 3, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.savefig('deep_learning_comparison.png')
plt.show()

print("Deep learning edge detection comparison completed!")
print("Note: This is using an untrained model for demonstration purposes.")
print("In a real implementation, you would train the model on the BSDS500 dataset.")