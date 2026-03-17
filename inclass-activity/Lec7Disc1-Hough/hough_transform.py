import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the original image
img = cv2.imread("images/beshik_room.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For final display
output_img = img_rgb.copy()

# 2. Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Apply stronger blur to reduce noise specifically to help circle detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# 4. Edge Detection (Canny)
edges = cv2.Canny(blurred, 50, 150)

# 5. Hough Line Transform (Probabilistic)
# Adjust thresholds: minLineLength filters out small noise lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw detected lines in Red
        cv2.line(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 6. Hough Circle Transform
# param1 = upper Canny threshold; param2 = accumulator threshold (higher=stricter, fewer false circles)
circles = cv2.HoughCircles(
    blurred, 
    cv2.HOUGH_GRADIENT, 
    dp=1, 
    minDist=50, 
    param1=50, 
    param2=30, 
    minRadius=10, 
    maxRadius=100
    # param2=60,
    # minRadius=20,
    # maxRadius=200
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # Draw circle outer edge in Green
        cv2.circle(output_img, center, radius, (0, 255, 0), 3)
        # Draw circle center in Blue
        cv2.circle(output_img, center, 2, (0, 0, 255), 3)

# 7. Visualization
plt.figure(figsize=(15, 10))

# Show step-by-step processing
plt.subplot(131)
plt.imshow(gray, cmap='gray')
plt.title('1. Grayscale')
plt.axis('off')

plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('2. Canny Edges')
plt.axis('off')

plt.subplot(133)
plt.imshow(output_img)
plt.title('3. Detected Lines (Red) & Circles (Green)')
plt.axis('off')

plt.tight_layout()
plt.savefig("output/hough_transform_result.jpg", bbox_inches='tight')
