import cv2
import matplotlib.pyplot as plt

# Load images in color (RGB)
img1 = plt.imread("images/su27l.jpeg")
img2 = plt.imread("images/su27u.jpeg")

# Create ORB detector
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 40 matches (flags=2 hides unmatched keypoints)
result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], None, flags=2)

# Display and save the output
plt.figure(figsize=(12, 6))
plt.imshow(result)
plt.title("ORB Keypoint Matching: Top 40 Matches")
plt.axis("off")
plt.savefig("output/rotated_su27_matches.jpg", bbox_inches='tight')
