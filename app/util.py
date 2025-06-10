import hashlib
from pathlib import Path
from fastapi import UploadFile, HTTPException

def get_file_extension(filename: str) -> str:
    """Extracts the file extension from a filename."""
    return filename.split(".")[-1].lower()

async def generate_filename(file: UploadFile) -> str:
    """Generates a unique filename based on the file's content hash."""
    file_bytes = await file.read()
    await file.seek(0)  # Reset file pointer to the beginning

    hash_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    extension = get_file_extension(file.filename)
    return f"{hash_id}.{extension}"

def validate_file_type(file: UploadFile):
    """Validates the content-type and extension of the uploaded file."""
    # These could also be moved to your config for more flexibility
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
    SUPPORTED_VIDEO_TYPES = ["video/mp4"]

    content_type = file.content_type
    extension = get_file_extension(file.filename)

    is_image = content_type in SUPPORTED_IMAGE_TYPES and extension in ["jpg", "jpeg", "png"]
    is_video = content_type in SUPPORTED_VIDEO_TYPES and extension in ["mp4"]

    if not (is_image or is_video):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Received content-type: '{content_type}'."
        )
