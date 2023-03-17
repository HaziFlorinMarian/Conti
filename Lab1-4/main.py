# OpenCV library
import cv2
# Numpy
import numpy as np

# Get file from files or webcams
cam = cv2.VideoCapture("Lane Detection Test Video 01.mp4")

# Lane lines
left_top = left_bottom = right_top = right_bottom = (0, 0)

while True:
    # Returns whether cv2 was able to get a frame, and the actual frame (which is a Numpy array)
    ret, frame = cam.read()

    if ret is False:
        break

    # Get current frame size
    height, width, channels = frame.shape

    # Shrink the frame to be able to fit around 12 separate windows
    width, height = width // 4, height // 4

    # Change frame size
    resized_frame = cv2.resize(frame, (width, height))
    cv2.imshow('Default', resized_frame)

    # Convert the image to grayscale
    gray_frame = np.dot(resized_frame[:, :, 0:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    cv2.imshow('Gray', gray_frame)

    # Let's show only trapezoid
    trapezoid_mask = np.zeros_like(resized_frame)

    bottom_left_trapezoid = (0, height)
    bottom_right_trapezoid = (width, height)
    top_left_trapezoid = (width * 0.45, height * 0.75)
    top_right_trapezoid = (width * 0.55, height * 0.75)

    # Clockwise, starting with the bottom left corner
    trapezoid_bounds = np.array(
        [bottom_left_trapezoid, top_left_trapezoid, top_right_trapezoid, bottom_right_trapezoid], dtype=np.int32)

    # A combination between cv2.convexHull (computing the convex skeleton) + cv2.fillPoly (fill convex polygons)
    cv2.fillConvexPoly(trapezoid_mask, trapezoid_bounds, (255, 255, 255))

    gray_trapezoid_mask = cv2.cvtColor(trapezoid_mask, cv2.COLOR_BGR2GRAY)

    # Only the pixels in the region of interest will be retained in the resulting image, while all other pixels will
    # be set to 0
    trapezoid_gray_frame = cv2.bitwise_and(gray_frame, gray_trapezoid_mask)
    cv2.imshow('Trapezoid', trapezoid_gray_frame)

    # Get top-down view
    top_left = (0, 0)
    top_right = (width, 0)

    screen_bounds = np.array([bottom_left_trapezoid, top_left, top_right, bottom_right_trapezoid], dtype=np.float32)
    trapezoid_bounds = np.float32(trapezoid_bounds)

    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds, screen_bounds)
    stretched_frame = cv2.warpPerspective(trapezoid_gray_frame, magic_matrix, (width, height))

    # Adding a bit of blur
    blurred_frame = cv2.blur(stretched_frame, ksize=(7, 7))

    # Doing edge detection
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    filtered_vertical_frame = cv2.filter2D(np.float32(blurred_frame), -1, sobel_vertical)
    filtered_horizontal_frame = cv2.filter2D(np.float32(blurred_frame), -1, sobel_horizontal)

    filtered_final_frame = np.sqrt(np.square(filtered_vertical_frame) + np.square(filtered_horizontal_frame))

    # Get the absolute value returned as 8-bit
    filtered_final_frame = cv2.convertScaleAbs(filtered_final_frame)

    cv2.imshow('Filtered', filtered_final_frame)

    # Binarize the frame
    threshold = 75

    # cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique)
    ret, binarized_frame = cv2.threshold(filtered_final_frame, threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow('Binarized', binarized_frame)

    # [9] Get the coordinates of street markings on each side of the road
    copied_frame = binarized_frame.copy()

    # Compute the number of columns to set to black
    num_cols = int(width * 0.05)

    # Set the first and last X columns to black
    copied_frame[:, :num_cols] = 0
    copied_frame[:, -num_cols:] = 0

    cv2.imshow('Black margins', copied_frame)

    # Split the copied frame into left and right halves
    left_frame = copied_frame[:, :width // 2]
    right_frame = copied_frame[:, width // 2:]

    # White points on each side
    left_white_points = np.argwhere(left_frame == 255)
    right_white_points = np.argwhere(right_frame == 255)

    left_y = left_white_points[:, 0]
    left_x = left_white_points[:, 1]
    right_y = right_white_points[:, 0]
    right_x = right_white_points[:, 1]

    # Set the correct position for x in the original image
    right_x += (width // 2)

    # [10] Find the lines that detect the edges of the lane

    # Fit a polynomial to the left lane line
    left_fit = np.polyfit(left_x, left_y, 1)

    # Fit a polynomial to the right lane line
    right_fit = np.polyfit(right_x, right_y, 1)

    # Calculate the two endpoints for the left lane line
    left_top_y = 0
    left_top_x = int(np.polyval(left_fit, left_top_y))
    left_bottom_y = height - 1
    left_bottom_x = int(np.polyval(left_fit, left_bottom_y))

    # Calculate the two endpoints for the right lane line
    right_top_y = 0
    right_top_x = int(np.polyval(right_fit, right_top_y))
    right_bottom_y = height - 1
    right_bottom_x = int(np.polyval(right_fit, right_bottom_y))

    # Check if any of the calculated x values are bad
    if -1e8 <= left_top_x <= 1e8:
        left_top = (left_top_x, left_top_y)
    if -1e8 <= left_bottom_x <= 1e8:
        left_bottom = (left_bottom_x, left_bottom_y)

    if -1e8 <= right_top_x <= 1e8:
        right_top = (right_top_x, right_top_y)
    if -1e8 <= right_bottom_x <= 1e8:
        right_bottom = (right_bottom_x, right_bottom_y)

    # Middle screen line
    cv2.line(binarized_frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

    # Draw the left lane line
    cv2.line(binarized_frame, left_top, left_bottom, (200, 0, 0), 3)

    # Draw the right lane line
    cv2.line(binarized_frame, right_top, right_bottom, (100, 0, 0), 3)

    cv2.imshow('Lines', binarized_frame)

    # cv2.waitKey(n) waits n ms for a key to be pressed and returns the code of that key
    # cv2.waitKey(n) & 0xFF gives the ASCII code of the letter (so the if is executed when we press "q").
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
