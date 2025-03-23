import cv2
import face_recognition
import math

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np

import secrets

class FalseLight:
    def __init__(self):
        self._model = None

    def loadModel(self, modelPath):
        self._model = tf.keras.models.load_model(modelPath)

    def makePrediction(self, face):
        x = img_to_array(face)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = self._model.predict(x, verbose=0)
        
        prob_fake = pred[0][0]
        prob_real = 1 - prob_fake
        realScore = float(f"{prob_real * 100:.2f}")
        fakeScore = float(f"{prob_fake * 100:.2f}")
    
        return [realScore, fakeScore]

    def getFaceLocations(self, img):
        faceLocs = face_recognition.face_locations(img)
        return faceLocs

    def extractFaces(self, faceLocs, img):
        faces = []
        for i, faceLoc in enumerate(faceLocs):
            top, right, bottom, left = faceLoc
            faceOnly = img[top:bottom, left:right]
            faces.append(faceOnly)
        return faces
    
    def processFace(self, face):
        resized = cv2.resize(face, (224, 224))
        colorCorrected = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        return colorCorrected

    def resizeToThumb(self, item):
        resized = cv2.resize(item, (64, 64))
        colorCorrected = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return colorCorrected

    def loadImage(self, imagePath):
        img = face_recognition.load_image_file(imagePath)
        return img

    def checkImage(self, mode, v, saveMode):
        img = None
        videoModeReturn = []
        if mode == "video":
            img = v
        else:
            img = self.loadImage(v)

        faceLocs = self.getFaceLocations(img)
        if faceLocs:
            faces = self.extractFaces(faceLocs, img)
            processedFaces = []
            for face in faces:
                processedFace = self.processFace(face)
                processedFaces.append(processedFace)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if mode == "video":
                for i in range(len(processedFaces)):
                    bbox = faceLocs[i]
                    real, fake = self.makePrediction(processedFaces[i])
                    thumb = self.resizeToThumb(processedFaces[i])
                    videoModeReturn.append([thumb, real, fake])
                return videoModeReturn
            else:
                for i in range(len(processedFaces)):
                    bbox = faceLocs[i]
                    print(bbox)

                    startPoint = (bbox[3], bbox[0])
                    endPoint = (bbox[1], bbox[2])
                    cv2.rectangle(img, startPoint, endPoint, (0, 0, 255), 2)

                    real, fake = self.makePrediction(processedFaces[i])

                    text = f"Fake: {fake}"
                    org = (bbox[3], bbox[0] - 10)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    color = (0, 0, 255)
                    thickness = 2
                    line_type = cv2.LINE_AA
                    cv2.putText(img, text, org, font, font_scale, color, thickness, line_type)

                if saveMode == True:
                    random_name = secrets.token_hex(8)
                    savedFilePath = f"processed/{random_name}.jpg"
                    cv2.imwrite(savedFilePath, img)
                    return [savedFilePath, real, fake]
                else:
                    cv2.imshow("Resized Face", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            print("No Face Detected")

    def loadVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Error: Could not open video.")
        
        return cap
    
    def processVideoIntoSegments(self, cap):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()

        num_parts = 10
        frames_to_capture = []
        interval = total_frames / num_parts
        frame_indices = [math.floor(i * interval) for i in range(num_parts)]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_to_capture.append(frame)
            else:
                print(f"Invalid Frame")

        cap.release()

        return frames_to_capture

    def create_grid_display(self, images, rows=2, cols=5, img_size=(64, 64), text="My Image Grid"):
        img_h, img_w = img_size
        text_area = 30
        total_height = rows * img_h + text_area
        total_width = cols * img_w
        
        display_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            img = cv2.resize(img, (img_w, img_h))

            y_start = r * img_h
            y_end = y_start + img_h
            x_start = c * img_w
            x_end = x_start + img_w

            display_image[y_start:y_end, x_start:x_end] = img

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (0, 0, 0)
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        text_x = (total_width - text_width) // 2
        text_y = rows * img_h + (text_area // 2) + (text_height // 2)

        cv2.putText(display_image, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        return display_image

    
    def checkVideo(self, videoPath, resolutionDownScale=2):
        cap = self.loadVideo(videoPath)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            orig_height, orig_width = frame.shape[:2]
            new_width = orig_width // resolutionDownScale
            new_height = orig_height // resolutionDownScale
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            faceLocs = self.getFaceLocations(rgb_frame)
        
            for faceLoc in faceLocs:
                top, right, bottom, left = faceLoc

                face_roi = rgb_frame[top:bottom, left:right]
                processed_face = self.processFace(face_roi)
                
                real, fake = self.makePrediction(processed_face)
                
                cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 0, 255), 2)
                
                text = f"Real: {real:.2f}%, Fake: {fake:.2f}%"
                text_y = top - 10 if top - 10 > 10 else top + 20
                cv2.putText(frame_resized, text, (left-170, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
            cv2.imshow("Video", frame_resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



