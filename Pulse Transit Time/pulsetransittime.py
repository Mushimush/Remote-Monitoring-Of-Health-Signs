import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


# Webcam Parameters
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)
    fps = webcam.get(cv2.CAP_PROP_FPS)
    realWidth = 640
    realHeight = 480
    videoWidth = 160
    videoHeight = 120
    videoWidth2 = 160
    videoHeight2 = 120
    videoChannels = 3
    videoFrameRate = 15
    detectionWidth = 160
    detectionHeight = 120
    webcam.set(3, realWidth)
    webcam.set(4, realHeight)


# Define the gap between the two detection frames
gap = 50

# Calculate the top-left and bottom-right corners of the original detection frame
top_left = (realWidth//2 - detectionWidth//2,
            realHeight//2 - detectionHeight//2)
bottom_right = (realWidth//2 + detectionWidth//2,
                realHeight//2 + detectionHeight//2)

# Calculate the top-left and bottom-right corners of the new detection frame
top_left2 = (bottom_right[0] + gap, top_left[1])
bottom_right2 = (top_left2[0] + detectionWidth, bottom_right[1])


# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros(
    (bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

firstFrame2 = np.zeros((videoHeight2, videoWidth2, videoChannels))
firstGauss2 = buildGauss(firstFrame2, levels+1)[levels]
videoGauss2 = np.zeros(
    (bufferSize, firstGauss2.shape[0], firstGauss2.shape[1], videoChannels))
fourierTransformAvg2 = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

bpmCalculationFrequency2 = 15
bpmBufferIndex2 = 0
bpmBufferSize2 = 10
bpmBuffer2 = np.zeros((bpmBufferSize))

bpmcount = []
bpmcount2 = []
bpmtest = []
time_counter = 0
times = []
mag_array = []
i = 0

while (True):
    ret, frame = webcam.read()
    if ret == False:
        break

    if len(sys.argv) != 2:
        originalFrame = frame.copy()
    detectionFrame = frame[realHeight//2 - detectionHeight//2:realHeight//2 + detectionHeight//2,
                           realWidth//2 - detectionWidth//2:realWidth//2 + detectionWidth//2, :]
    # Extract the new detection frame from the original frame
    detectionFrame2 = frame[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0], :]

    # Construct Gaussian Pyramid
    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
    fourierTransform = np.fft.fft(videoGauss, axis=0)

    videoGauss2[bufferIndex] = buildGauss(detectionFrame2, levels+1)[levels]
    fourierTransform2 = np.fft.fft(videoGauss2, axis=0)

    # Bandpass Filter
    fourierTransform[mask == False] = 0
    fourierTransform2[mask == False] = 0
    
    # Compute the magnitude of the Fourier Transform for both signals
    mag = np.abs(fourierTransform[:,0,0,0])
    time = np.arange(len(mag)) / fps

    mag2 = np.abs(fourierTransform2[:,0,0,0])
    time2 = np.arange(len(mag2)) / fps

    # Calculate the cross-correlation between the two signals
    cross_corr = np.correlate(mag, mag2, mode='full')

    # Find the time lag at which the cross-correlation is maximal
    lag = np.argmax(cross_corr) - len(mag) + 1  # subtract len(mag) to account for zero padding, add 1 to account for indexing from 0
    
    
    if lag > 0 :
        # Print the time lag in seconds
        print("Time lag: {:.3f} s".format(lag / fps))
        # Plot the two signals and the cross-correlation
        fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))

        ax[0].plot(time, mag, label='Signal 1')
        ax[1].plot(time2, mag2, label='Signal 2')
        ax[2].plot(np.arange(len(cross_corr)) / fps - len(mag) / fps + 1 / fps, cross_corr, label='Cross-correlation')
        ax[2].axvline(x=lag / fps, color='r', linestyle='--', label='Max correlation lag')

        ax[0].set_ylabel('Magnitude')
        ax[1].set_ylabel('Magnitude')
        ax[2].set_xlabel('Time lag (s)')
        ax[2].set_ylabel('Cross-correlation')

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        plt.show()

    # Grab a Pulse
    if bufferIndex % bpmCalculationFrequency == 0:
        i = i + 1
        for buf in range(bufferSize):
            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            fourierTransformAvg2[buf] = np.real(fourierTransform2[buf]).mean()
        hz = frequencies[np.argmax(fourierTransformAvg)]
        hz2 = frequencies[np.argmax(fourierTransformAvg2)]
        bpm = 60.0 * hz
        bpm2 = 60.0 * hz2
        bpmcount.append(bpm)
        bpmcount2.append(bpm2)
        bpmBuffer[bpmBufferIndex] = bpm
        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

    # Amplify
    filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
    filtered = filtered * alpha

    filtered2 = np.real(np.fft.ifft(fourierTransform2, axis=0))
    filtered2 = filtered2 * alpha

    # Reconstruct Resulting Frame
    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
    outputFrame = detectionFrame + filteredFrame
    outputFrame = cv2.convertScaleAbs(outputFrame)

    filteredFrame2 = reconstructFrame(filtered2, bufferIndex, levels)
    outputFrame2 = detectionFrame2 + filteredFrame2
    outputFrame2 = cv2.convertScaleAbs(outputFrame2)

    bufferIndex = (bufferIndex + 1) % bufferSize

    frame[realHeight//2 - detectionHeight//2:realHeight//2 + detectionHeight//2,
          realWidth//2 - detectionWidth//2:realWidth//2 + detectionWidth//2, :] = outputFrame

    frame[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0], :] = outputFrame2

    # Calculate the coordinates of the top-left and bottom-right corners
    # of the detection rectangle
    top_left = (int((realWidth - detectionWidth) / 2),
                int((realHeight - detectionHeight) / 2))
    bottom_right = (int((realWidth + detectionWidth) / 2),
                    int((realHeight + detectionHeight) / 2))

    # Draw the detection rectangle on the frame
    cv2.rectangle(frame, top_left, bottom_right, boxColor, boxWeight)

    # Calculate the coordinates of the top-left and bottom-right corners
    # of the detection rectangle for detectionFrame2
    offset = 50  # example offset value
    top_left2 = (bottom_right[0] + offset, top_left[1])
    bottom_right2 = (top_left2[0] + detectionWidth, bottom_right[1])

    # Draw the detection rectangle for detectionFrame2 on the frame
    cv2.rectangle(frame, top_left2, bottom_right2, boxColor, boxWeight)
    if i > bpmBufferSize:
        cv2.putText(frame, "Calculating" ,
                    bpmTextLocation, font, fontScale, fontColor, lineType)
    else:
        cv2.putText(frame, "Calibrating", loadingTextLocation,
                    font, fontScale, fontColor, lineType)

    if len(sys.argv) != 2:
        cv2.imshow("Webcam Heart Rate Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()