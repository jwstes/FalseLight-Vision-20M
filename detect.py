import argparse
import cv2
from FalseLight import FalseLight

def detect_image(model, image_path, save_mode=False):
    result = model.checkImage("image", image_path, save_mode)
    if save_mode:
        print("Processed image saved to:", result[0])
    else:
        print("Image processing complete. Close the window to exit.")

def detect_video(model, video_path):
    model.checkVideo(video_path, resolutionDownScale=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FalseLight Detection Demo")
    parser.add_argument('--mode', choices=['image', 'video'], required=True,
                        help="Detection mode: 'image' for static images or 'video' for video streams")
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the input image or video file")
    parser.add_argument('--save', action='store_true',
                        help="If set, the processed image will be saved to disk (only applicable in image mode)")
    args = parser.parse_args()

    fl = FalseLight()
    model_path = "FalseLight Vision 20M.h5"
    fl.loadModel(model_path)
    print(f"Loaded model from {model_path}")

    if args.mode == 'image':
        detect_image(fl, args.input, args.save)
    elif args.mode == 'video':
        detect_video(fl, args.input)


#python demo_detection.py --mode image --input path/to/your/image.jpg --save
#python demo_detection.py --mode video --input path/to/your/video.mp4
