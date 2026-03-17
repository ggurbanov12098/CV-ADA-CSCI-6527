# Hough Transform Activity

This project demonstrates the application of the **Hough Transform** algorithm to detect distinct mathematical shapes (specifically circles and straight lines) within a real-world image. 

The source image is located at `images/beshik_room.jpg`.

## Step-by-Step Logic

The `hough_transform.py` script computes the transform cleanly and sequentially from scratch:

1. **Load Image**: The original image is read into memory.
2. **Grayscale Conversion**: `cv2.cvtColor()` strips all color, converting the image to intensity values. Edge detection algorithms require this because they look for stark jumps in light/dark pixel intensity.
3. **Noise Reduction**: We apply a slight Gaussian Blur `cv2.GaussianBlur(gray, (5, 5), 0)`. The Hough transform is incredibly sensitive; without blurring, it will detect every single microscopic texture change as a "line" or "circle".
4. **Edge Detection**: `cv2.Canny(blurred, 50, 150)` pulls out the hard edges of the objects in the room. This maps out the geometric skeleton of the image.
5. **Hough Line Transform**: Using the Probabilistic Hough Line Transform (`cv2.HoughLinesP()`), we search the Canny edge map for straight lines matching mathematical parameters. Adjusting the `threshold` and `minLineLength` values filters out small "noise" lines, leaving only structural straight edges. The lines are overlaid onto the image in **Red**.
6. **Hough Circle Transform**: We scan for circular arcs using `cv2.HoughCircles()`. Parameters like `minRadius` and `maxRadius` enforce exactly what sizes define a true circle in the context of our room. The outline is drawn in **Green**, and the center point is mapped in **Blue**.

## Output

The result is saved in `output/hough_transform_result.jpg`. 

You can visually see the side-by-side processing mapping the original grayscale image to the stripped Canny edges, and finally to the correctly isolated vectors and arcs. The red lines map straight wall and furniture borders, while the green circles successfully identify rounded objects in the room.

## Setup and Running

Requires OpenCV and Matplotlib in your Python environment:
```bash
pip install opencv-python matplotlib numpy
```

To run:
```bash
python hough_transform.py
```
