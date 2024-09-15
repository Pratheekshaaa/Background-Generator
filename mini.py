import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FPS, 60)

# Initialize the SelfiSegmentation module
segmentor = SelfiSegmentation(model=1)

# Load the images using forward slashes
bg_path = "C:/Cg project/img1.jpeg"
imgBg = cv2.imread(bg_path)
if imgBg is None:
    raise ValueError(f"Background image at {bg_path} not found.")
imgBg = cv2.resize(imgBg, (640, 480))

ref1_path = "C:/Cg project/img2.jpeg"
imgRef1 = cv2.imread(ref1_path)
if imgRef1 is None:
    raise ValueError(f"Reference image 1 at {ref1_path} not found.")
imgRef1 = cv2.resize(imgRef1, (640, 480))

ref2_path = "C:/Cg project/img3.jpeg"
imgRef2 = cv2.imread(ref2_path)
if imgRef2 is None:
    raise ValueError(f"Reference image 2 at {ref2_path} not found.")
imgRef2 = cv2.resize(imgRef2, (640, 480))

ref3_path = "C:/Cg project/img4.jpeg"
imgRef3 = cv2.imread(ref3_path)
if imgRef3 is None:
    raise ValueError(f"Reference image 3 at {ref3_path} not found.")
imgRef3 = cv2.resize(imgRef3, (640, 480))

# List of images and their titles
images = [imgBg, imgRef1, imgRef2, imgRef3]
image_titles = ["Background Image", "Reference Image 1", "Reference Image 2", "Reference Image 3"]
current_index = 0

# FPS calculation variables
pTime = 0
fps_list = []

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam. Exiting...")
        break
    
    # Apply self-segmentation
    imgOut = segmentor.removeBG(img, images[current_index])
    
    # Apply edge enhancement
    edges = cv2.Canny(imgOut, 900, 900)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    imgOut = cv2.addWeighted(imgOut, 0.9, edges_colored, 0.9, 0)
    
    # Ensure imgOut is the same size as img
    imgOut = cv2.resize(imgOut, (img.shape[1], img.shape[0]))
    
    # Ensure both images are of the same type
    if imgOut.dtype != img.dtype:
        imgOut = imgOut.astype(img.dtype)
    
    # Stack images for display (original and output image)
    imgStacked = cv2.hconcat([img, imgOut])
    
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    fps_list.append(fps)
    pTime = cTime
    
    # Smooth FPS
    if len(fps_list) > 10:  # Use the last 10 FPS values for smoothing
        fps_list.pop(0)
    avg_fps = np.mean(fps_list)
    
    # Display FPS and image title on the image
    cv2.putText(imgStacked, f'FPS: {int(avg_fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(imgStacked, image_titles[current_index], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the stacked images
    cv2.imshow("Segmented Image", imgStacked)
    
    # Check for key presses
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):  # Press 'n' for next image
        current_index = (current_index + 1) % len(images)

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
