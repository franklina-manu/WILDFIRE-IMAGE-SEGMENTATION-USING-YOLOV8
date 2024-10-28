import shutil
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import cv2
from fastapi.staticfiles import StaticFiles
from inference_sdk import InferenceHTTPClient
from contextlib import asynccontextmanager
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    yield
    # Shutdown event
    if DETECTION_DIR.exists():
        shutil.rmtree(DETECTION_DIR)  # Clean up the entire folder on server shutdown


app = FastAPI(
    title="FIRE DETECTION",
    description="",
    version="0.1.0",
    lifespan=lifespan,  # Attach lifespan handler
)

origins = [
    "http://localhost:5500",  # Update with your frontend URL
    "https://accomodation-link.vercel.app",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", api_key="0gvK5Tu6B6V2daODNXoP"
)

# Directory to save detection images temporarily
DETECTION_DIR = Path("detection/")
DETECTION_DIR.mkdir(exist_ok=True)

# Serve static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve HTML page
@app.get("/", response_class=HTMLResponse)
async def get_root():
    # Return the HTML file as response
    html_file_path = Path("ui/index.html")
    return html_file_path.read_text()


@app.post("/detect-fire/")
async def detect_fire(file: UploadFile):
    # Save the uploaded file
    file_path = DETECTION_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Call the inference API
    result = CLIENT.infer(str(file_path), model_id="fire-smoke-yolov8/1")

    image = cv2.imread(str(file_path))

    # Flag to check if fire is detected
    fire_detected = False

    # Process the result and draw bounding boxes
    for prediction in result["predictions"]:
        x_center = prediction["x"]
        y_center = prediction["y"]
        width = prediction["width"]
        height = prediction["height"]
        confidence = prediction["confidence"] * 100  # Convert to percentage
        label = prediction["class"]

        # Convert center x, y, width, and height to top-left and bottom-right corners
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw rectangle (bounding box)
        color = (
            (0, 0, 255) if label == "Fire" else (255, 0, 0)
        )  # Red for fire, blue for smoke
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add label and confidence
        label_text = f"{label} {confidence:.2f}%"
        cv2.putText(
            image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

        # Set flag if fire is detected
        if label == "Fire":
            fire_detected = True

    # If no fire detected, add text to the image
    if not fire_detected:
        no_fire_text = "No Fire Detected"
        cv2.putText(
            image, no_fire_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # Save the modified image
    detected_img_path = DETECTION_DIR / f"detected_{file.filename}"
    cv2.imwrite(str(detected_img_path), image)

    # Return the detected image as a downloadable file
    return FileResponse(
        str(detected_img_path),
        media_type="image/png",
        filename=f"detected_{file.filename}",
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        use_colors=True,
    )
