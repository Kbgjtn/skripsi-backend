import time
import aiofiles

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, HTTPException, File, Query

from .config import get_settings
from . import util
from . import predict

settings = get_settings()

app = FastAPI(lifespan=predict.lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/assets", 
    StaticFiles(directory=settings.PREDICTIONS_DIRECTORY), 
    name="assets"
)

@app.get("/")
def read_root():
    return {
            "code": 200,
            "timestamp": int(time.time()),
            "message": "Image and Video Prediction REST API"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    util.validate_file_type(file)

    try:
        new_filename = await util.generate_filename(file)
        save_path = settings.UPLOAD_DIRECTORY / new_filename

        async with aiofiles.open(save_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(
                status_code= 500,
                detail=f"There was an error uploading the file: {e}"
            )
    finally:
        await file.close()

    return JSONResponse(
            status_code=201,
            content={
                "messsage": "File uploaded successfully",
                "filename": new_filename
            }
        )

@app.get("/predict/{filename}")
async def predict(
        filename: str,
        conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold for detection."),
        imgsz: int = Query(640, gt=0, description="Image size for the model input (e.g., 640)."),
        speed_factor: float = Query(2.0, ge=0.1, le=10.0, description="Playback speed factor. >1 for slower, <1 for faster (e.g., 2.0 is half speed).")
):
    """
    Performs prediction on an uploaded file. Uses the appropriate model
    based on the file type (image or video).
    
    You can adjust prediction parameters using query strings.
    - `conf`: Confidence threshold (for videos)
    - `imgsz`: Image size for model input
    - `speed_factor`: Adjust video playback speed. >1 for slower, <1 for faster.
    
    Example: `/predict/some_video.mp4?speed_factor=4.0`
    """

    file_path = settings.UPLOAD_DIRECTORY / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    extension = util.get_file_extension(filename)
    
    try:
        if extension in ["jpg", "jpeg", "png"]:
            if not predict.models.get("classifier"):
                raise HTTPException(status_code=500, detail="CNN model is not available.")
            
            results = predict.predict_with_classifier(file_path, imgsz)

        elif extension in ["mp4"]:
            if not predict.models.get("detection"):
                    raise HTTPException(status_code=500, detail="YOLO detection model is not available.")

            results = predict.predict_with_detection(file_path, conf, imgsz, speed_factor)
        else:
            # This case should technically not be reached due to upload validation
            raise HTTPException(status_code=400, detail="Unsupported file type for prediction")

    except Exception as e:
        # Catch potential errors from the model inference step
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")

    return JSONResponse(content={"filename": filename, "results": results})
