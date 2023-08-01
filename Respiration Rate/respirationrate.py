import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Start the camera
cap = cv2.VideoCapture(0)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Checking the difference for every frame will be very computationally expensive, very lag, we compute the difference every 15 frames, every 0.5 seconds
frames_to_skip = frame_rate * 0.5

prevs_frame = None

# Initialize the variables
counter = 0
time_counter = 0
cycles = 0
sign_changes = 0
prev_gradient = 0
respiration_rate = 0
respiration_rate_avg = 0
time_for_each_rr = 0
time_elapsed = 0
movingaverage_values = []
time_values = []
respiration_rate_list = []
detected_cycles = []


# Threshold values
tresh = 30
maxval = 255


# Start the timer, once webcam is on
start_time = time.time()


# Initialize the webcam
realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
cap.set(3, realWidth)
cap.set(4, realHeight)
boxColor = (0, 255, 0)
boxWeight = 3

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3


# Moving average function
def moving_average(data, window_size):
    window = np.ones(int(np.ceil(window_size)))/window_size
    return np.convolve(data, window, 'same')


while True:
    ret, frame = cap.read()
    detectionFrame = frame[realHeight // 2:,
                           realWidth // 10:realWidth * 9 // 10, :]
    cv2.rectangle(frame, (realWidth // 10, realHeight // 2),
                  (realWidth * 9 // 10, realHeight), boxColor, boxWeight)
    gray = cv2.cvtColor(detectionFrame, cv2.COLOR_BGR2GRAY)

    # If the previous frame is available, find the difference between the frames
    if prevs_frame is not None and counter % frames_to_skip == 0:
        imgdiff = cv2.subtract(gray, prevs_frame)
    else:
        imgdiff = gray
    if counter % frames_to_skip == 0:
        prevs_frame = gray
        counter = 0
    else:
        counter += 1

    # Apply a threshold to the difference in frames
    retval, threshold = cv2.threshold(
        imgdiff, tresh, maxval, cv2.THRESH_BINARY)
    cv2.imshow('imgdiff', threshold)

    # The threshold image will be binary, sum up all ones
    pixel_sum = np.sum(threshold)

    # Append the sums to a list, and use a moving average to smooth out the peaks
    movingaverage_values.append(pixel_sum)
    time_values.append(time_counter)
    time_counter += 1
    moving_average_sum = moving_average(movingaverage_values, 4)

    # Check for 4 sign changes, if present, count it as one respiration cycle
    for i in range(1, len(moving_average_sum)):
        gradient = moving_average_sum[i] - moving_average_sum[i-1]
        if (prev_gradient < 0 and gradient > 0) or (prev_gradient > 0 and gradient < 0):
            sign_changes += 1
        if sign_changes == 4:
            if i not in detected_cycles:
                time_for_each_rr = time.time() - start_time
                start_time = time.time()
                print(time_for_each_rr)
                if time_for_each_rr >= 1.3:
                    time_elapsed += time_for_each_rr
                    cycles += 1
                    respiration_rate = cycles/time_elapsed * 60
                    respiration_rate_list.append(respiration_rate)
                    respiration_rate_avg = sum(
                        respiration_rate_list)/len(respiration_rate_list)
                detected_cycles.append(i)
            sign_changes = 0
        prev_gradient = gradient

    # Display the readings
    if respiration_rate != 0:
        cv2.putText(frame, "Respiration Rate: %d" % respiration_rate_avg,
                    bpmTextLocation, font, fontScale, fontColor, lineType)
    else:
        cv2.putText(frame, "Calculating Respiration Rate...", loadingTextLocation,
                    font, fontScale, fontColor, lineType)
    cv2.rectangle(frame, (realWidth // 10, realHeight // 2),
                  (realWidth * 9 // 10, realHeight), boxColor, boxWeight)
    cv2.imshow('Camera', frame)
    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(cycles)
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()