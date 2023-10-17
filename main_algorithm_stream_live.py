# Importing the required packages
import numpy as np
import cv2
import os
import psutil
import time

# Initialize variables for calculating average memory usage and process time
total_memory_usage = 0
total_process_time = 0
process_time = 0
k = 10
global cap 
cap = cv2.VideoCapture(0)

for i in range(k):
    # Color selection (HSL)
    def convert_hsl(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def HSL_color_selection(image):
        # Convert the input image to HSL
        converted_image = convert_hsl(image)

        # White color mask
        lower_threshold = np.uint8([0, 200, 0])
        upper_threshold = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

        # Yellow color mask
        lower_threshold = np.uint8([10, 0, 100])
        upper_threshold = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

        # Combine white and yellow masks
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        return masked_image

    # Canny edge detection
    def gray_scale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def gaussian_smoothing(image, kernel_size=13):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def canny_detector(image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    # Region of interest
    def region_selection(image):
        mask = np.zeros_like(image)
        # Defining a 3-channel or 1-channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        # Define the vertices of the polygon dynamically based on the image dimensions
        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    # Hough transform
    def hough_transform(image):
        """
        Determine and cut the region of interest in the input image.
            Parameters:
                image: The output of a Canny transform.
        """
        rho = 1              # Distance resolution of the accumulator in pixels.
        theta = np.pi/180    # Angle resolution of the accumulator in radians.
        threshold = 20       # Only lines that are greater than threshold will be returned.
        minLineLength = 20   # Line segments shorter than that are rejected.
        maxLineGap = 300     # Maximum allowed gap between points on the same line to link them
        return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                               minLineLength=minLineLength, maxLineGap=maxLineGap)

    def draw_lines(image, lines, color=[0, 255, 0], thickness=2):
        """
        Draw lines onto the input image.
            Parameters:
                image: An np.array compatible with plt.imshow.
                lines: The lines we want to draw.
                color (Default = red): Line color.
                thickness (Default = 2): Line thickness.
        """
        image = np.copy(image)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        return image

    # Averaging and extrapolating the lane lines
    def average_slope_intercept(lines):
        """
        Find the slope and intercept of the left and right lanes of each image.
            Parameters:
                lines: The output lines from Hough Transform.
        """
        left_lines = []  # (slope, intercept)
        left_weights = []  # (length,)
        right_lines = []  # (slope, intercept)
        right_weights = []  # (length,)

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane

    def pixel_points(y1, y2, line):
        """
        Converts the slope and intercept of each line into pixel points.
            Parameters:
                y1: y-value of the line's starting point.
                y2: y-value of the line's end point.
                line: The slope and intercept of the line.
        """
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))

    def lane_lines(image, lines):
        """
        Create full length lines from pixel points.
            Parameters:
                image: The input test image.
                lines: The output lines from Hough Transform.
        """
        left_lane, right_lane = average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = pixel_points(y1, y2, left_lane)
        right_line = pixel_points(y1, y2, right_lane)
        return left_line, right_line

    def draw_lane_lines(image, lines, color=[0, 255, 0], thickness=12):
        """
        Draw lines onto the input image.
            Parameters:
                image: The input test image.
                lines: The output lines from Hough Transform.
                color (Default = red): Line color.
                thickness (Default = 12): Line thickness.
        """
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    def frame_processor(image):
        """
        Process the input frame to detect lane lines.
            Parameters:
                image: Single video frame.
        """
        color_select = HSL_color_selection(image)
        gray = gray_scale(color_select)
        smooth = gaussian_smoothing(gray)
        edges = canny_detector(smooth)
        region = region_selection(edges)
        hough = hough_transform(region)
        result = draw_lane_lines(image, lane_lines(image, hough))
        return result

    def process_video(output_video):
        global cap
        """
        Read input video stream and produce a video file with detected lane lines.
            Parameters:
                output_video: A video file with detected lane lines.
        """
        start_time = time.time()
          # Capture video from the default camera
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    try : 
        processed_frame = frame_processor(frame) 
        cv2.imshow('Lane Detection', processed_frame)
    except:
        cv2.imshow('Lane Detection', frame)
        
    # output.write(processed_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# output.release()
cv2.destroyAllWindows()