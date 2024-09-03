import cv2
import mediapipe as mp
import os
from gradio_client import Client
# from test_image_fusion import Test
# from test_image_fusion import Test
from test_image import Test
import numpy as np



from PIL import Image
import numpy as np
import cv2

# client = Client("https://tbvl-real-and-fake-face-detection.hf.space/--replicas/40d41jxhhx/")

data = 'faceswap'
dct = 'fft'


# testet = Test(model_paths = [f"weights/{data}-hh-best_model.pth",
#                             f"weights/{data}-fft-best_model.pth"],
#                             multi_modal = ['hh', 'fft'])

testet = Test(model_path =f"weights/{data}-hh-best_model.pth",
                            multi_modal ='hh')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.35)

# Create a directory to save the cropped face images if it does not exist
save_dir = "cropped_faces"
os.makedirs(save_dir, exist_ok=True)

# def detect_and_label_faces(image_path):


# Function to crop faces from a video and save them as images
# def crop_faces_from_video(video_path):
#     # Read the video
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # Define the codec and create VideoWriter object
#     out = cv2.VideoWriter(f'output_{real}_{data}_fusion.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
# Convert PIL Image to NumPy array for OpenCV
def pil_to_opencv(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

# Convert OpenCV NumPy array to PIL Image
def opencv_to_pil(opencv_image):
    # Convert BGR to RGB
    pil_image = Image.fromarray(opencv_image[:, :, ::-1])
    return pil_image




def detect_and_label_faces(frame):
    frame = pil_to_opencv(frame)


    print(type(frame))
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Perform face detection
    results = face_detection.process(frame_rgb)

    # If faces are detected, crop and save each face as an image
    if results.detections:
        for face_count,detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # Crop the face region and make sure the bounding box is within the frame dimensions
            crop_img = frame[max(0, y):min(ih, y+h), max(0, x):min(iw, x+w)]
            if crop_img.size > 0:
                face_filename = os.path.join(save_dir, f'face_{face_count}.jpg')
                cv2.imwrite(face_filename, crop_img)

                label = testet.testimage(face_filename)

                if os.path.exists(face_filename):
                    os.remove(face_filename)

                color = (0, 0, 255) if label == 'fake' else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return opencv_to_pil(frame)

