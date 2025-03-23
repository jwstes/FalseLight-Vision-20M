from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import secrets
from FalseLight import FalseLight
import os
import cv2
import numpy as np

app = FastAPI()

TEMPLATES_DIR = Path(__file__).parent / "templates"

UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = Path(__file__).parent / "processed"
app.mount("/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/static", StaticFiles(directory="static"), name="static")

modelName = 'FalseLight Vision 20M.h5'
lunaris = FalseLight()
lunaris.loadModel(modelName)

print(f"Using Model: {modelName}")


@app.post("/upload")
async def upload_photo(photo: UploadFile = File(...)):
    file_extension = Path(photo.filename).suffix
    random_name = secrets.token_hex(8)
    file_name = f"{random_name}{file_extension}"
    file_path = UPLOADS_DIR / file_name

    try:
        with open(file_path, "wb") as f:
            f.write(await photo.read())
        
        results = lunaris.checkImage("image", f"uploads/{file_name}", True)
        print(results)

        os.remove(file_path)

        return JSONResponse(content={
            "image_path": f"/processed/{Path(results[0]).name}",
            "real" : results[1],
            "fake" : results[2]
        }, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    file_extension = Path(video.filename).suffix
    random_name = secrets.token_hex(8)
    file_name = f"{random_name}{file_extension}"
    file_path = UPLOADS_DIR / file_name

    try:
        with open(file_path, "wb") as f:
            f.write(await video.read())
        
        video_feed_url = f"/video_feed?video={file_name}"
        return JSONResponse(content={"video_feed": video_feed_url}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")


def generate_video_stream(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_height, orig_width = frame.shape[:2]
        new_width = orig_width // 5
        new_height = orig_height // 5
        frame_resized = cv2.resize(frame, (new_width, new_height))
        
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        faceLocs = lunaris.getFaceLocations(rgb_frame)
        for faceLoc in faceLocs:
            top, right, bottom, left = faceLoc
            face_roi = rgb_frame[top:bottom, left:right]
            processed_face = lunaris.processFace(face_roi)
            real, fake = lunaris.makePrediction(processed_face)
            cv2.rectangle(frame_resized, (left, top), (right, bottom), (0, 0, 255), 2)
            text = f"R:{real:.2f} F:{fake:.2f}"
            text_y = top - 10 if top - 10 > 10 else top + 20
            cv2.putText(frame_resized, text, (left, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        ret2, buffer = cv2.imencode('.jpg', frame_resized)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.get("/video_feed")
async def video_feed(video: str):
    file_path = UPLOADS_DIR / video
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return StreamingResponse(generate_video_stream(str(file_path)),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/processed", response_class=HTMLResponse)
async def processed_page(image: str):
    processed_file = TEMPLATES_DIR / "processed.html"
    if not processed_file.exists():
        return HTMLResponse(content="<h1>File Not Found</h1>", status_code=404)
    
    html_content = processed_file.read_text().replace("{{ image_path }}", image)
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = TEMPLATES_DIR / "landing.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>File Not Found</h1>", status_code=404)


@app.get("/detector", response_class=HTMLResponse)
async def read_detector():
    index_file = TEMPLATES_DIR / "detector.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>File Not Found</h1>", status_code=404)