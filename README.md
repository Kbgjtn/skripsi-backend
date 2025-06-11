# Image and Video Prediction API

This is a robust, production-ready REST API built with FastAPI and powered by YOLO models. It provides a complete solution for uploading images and videos, performing machine learning predictions (classification for images, object detection for videos), and serving the labeled results.

The entire service is containerized with Docker and managed via Docker Compose, ensuring a reproducible and scalable environment.

---

## Core Features

- **File Upload**: Securely upload images (`.jpg`, `.png`) and videos (`.mp4`).
- **Machine Learning Predictions**:
  - **Image Classification**: Uses a YOLO classification model to identify the contents of an image.
  - **Video Object Detection**: Uses a YOLO object detection model to find and track objects frame-by-frame.
- **Dynamic Labeling**: Automatically generates new image and video files with predictions (class names, confidence scores, bounding boxes) drawn directly onto them.
- **Static File Serving**: Exposes the labeled media files through a dedicated `/assets` endpoint.
- **Configurable Environment**: All settings, including model paths, server ports, and supported file types, are managed through a `.env` file for easy configuration across different environments (dev, prod).
- **Production-Ready Setup**:
  - **Dockerized**: The entire application and its dependencies are containerized.
  - **Multi-Stage Dockerfile**: Creates a small, optimized, and secure final image.
  - **Docker Compose**: Simplifies setup and management of the service and its volumes.
  - **Non-Root User**: Runs the application as a non-root user inside the container for enhanced security.
  - **Named Volumes**: Ensures data (uploads, predictions, models) persists reliably.

---

## API Documentation

The API exposes three main endpoints.

### 1. `POST /upload`

Upload an image or video file to the service. The file is saved and a unique filename is returned, which is then used for prediction.

- **Method**: `POST`
- **Endpoint**: `/upload`
- **Body**: `multipart/form-data` with a single file field.
- **Supported Content-Types**: `image/jpeg`, `image/png`, `video/mp4` (configurable via `.env`).

#### Example Request (`curl`)

```bash
curl -X POST "[http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"

Example Success Response (201 Created)

{
  "filename": "a1b2c3d4e5_1678886400.jpg"
}
```

2. GET /predict/{filename}
   Perform a prediction on a previously uploaded file. The service automatically uses the correct model (classification for images, detection for videos) based on the file extension.

   - Method: GET
   - Endpoint: /predict/{filename}
   - Path Parameter:
     filename (string, required): The unique filename returned by the /upload endpoint.
   - Query Parameters:
     conf (float, optional, default: 0.25): Confidence threshold for object detection in videos.
     imgsz (int, optional, default: 640): Image size for model input.
     speed_factor (float, optional, default: 2.0): Playback speed modifier for output videos. >1 is slower, <1 is faster.

Example Request (curl)

- For an image:

```sh
curl -X GET "[http://127.0.0.1:8000/predict/a1b2c3d4e5_1678886400.jpg?imgsz=1280](http://127.0.0.1:8000/predict/a1b2c3d4e5_1678886400.jpg?imgsz=224)"
```

- For a video (slowing it down to quarter speed):

```sh
curl -X GET
"[http://127.0.0.1:8000/predict/f6g7h8i9j0_1678886500.mp4?conf=0.4&speed_factor=4.0](http://127.0.0.1:8000/predict/f6g7h8i9j0_1678886500.mp4?conf=0.4&speed_factor=4.0&imgsz=224)"
```

Example Success Response (200 OK)

- For an image:

```json
{
  "filename": "28b0273b7d6010af.png",
  "results": {
    "path": "assets/28b0273b7d6010af_predicted.jpg",
    "predictions": [
      {
        "class_name": "algal_spot",
        "confidence": 0.9996527433395386,
        "id": 1,
        "name": "Algal Spot",
        "slug": "algal_spot",
        "description": "Algal spot merupakan penyakit yang menyerang tanaman teh yang disebabkan oleh alga Cephaleuros virescens. Daun teh yang mengidap penyakit ini memiliki gejala berupa bercak yang menonjol, berbentuk bulat, berwarna oranye hingga cokelat tua yang tersebar di permukaan daun."
      },
      {
        "class_name": "helopeltis",
        "confidence": 0.00024160525936167687,
        "id": 5,
        "name": "Helopeltis",
        "slug": "helopeltis",
        "description": "Helopeltis merupakan sebuah serangga hama yang menjadi ancaman bagi berbagai tanaman budidaya di Asia, termasuk tanaman teh. Hama ini merusak tanaman inangnya dengan cara menghisap cairan, yang menghasilkan munculnya bercak coklat atau tusukan kecil pada permukaan tanaman"
      },
      {
        "class_name": "gray_blight",
        "confidence": 4.34454414062202e-5,
        "id": 3,
        "name": "Gray Blight",
        "slug": "gray_blight",
        "description": "Infeksi yang menimbulkan bercak abu-abu pada daun, biasanya menyerang daun-daun yang sudah tua."
      }
    ],
    "model": "YOLO Classifier"
  }
}
```

For a video:

```json
{
  "filename": "f32dc7a0cd77e655.mp4",
  "results": {
    "path": "assets/f32dc7a0cd77e655_predicted.mp4",
    "detections": [
      [
        {
          "box_coordinates": [
            2917.26708984375, 0.0, 3837.86279296875, 740.156982421875
          ],
          "confidence": 0.25878429412841797,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            284.9656066894531, 1472.3970947265625, 1475.37255859375, 2160.0
          ],
          "confidence": 0.25862830877304077,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            2886.787353515625, 0.0, 3838.641357421875, 724.7286987304688
          ],
          "confidence": 0.2710469663143158,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            0.3017316460609436, 0.0, 586.510986328125, 691.0223388671875
          ],
          "confidence": 0.2509526312351227,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            2894.89111328125, 0.0, 3838.8662109375, 723.3228759765625
          ],
          "confidence": 0.2937139868736267,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            2889.59619140625, 0.0, 3837.7509765625, 727.6832275390625
          ],
          "confidence": 0.28483179211616516,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            2879.1650390625, 0.0, 3838.232666015625, 726.5857543945312
          ],
          "confidence": 0.25069525837898254,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ],
      [
        {
          "box_coordinates": [
            2876.187744140625, 0.0, 3836.28515625, 740.77001953125
          ],
          "confidence": 0.25497981905937195,
          "class_id": 5,
          "class_name": "red-rust",
          "id": 8,
          "name": "Red Rust",
          "slug": "red_rust",
          "description": "Penyakit red rust merupakan penyakit pada tanaman teh yang disebabkan oleh alga Cephaleuros parasiticus Karst. Penyakit ini ditandai dengan munculnya bercak-bercak berwarna merah atau oranye pada permukaan daun."
        }
      ]
    ],
    "model": "YOLOv8 Detection"
  }
}
```

3. GET /assets/{filename}
   Serve a static file (image or video) from the assets directory. This is used to retrieve the labeled media generated by the /predict endpoint.

- Method: GET
- Endpoint: /assets/{filename}

Example Request (In Browser or curl)

#### To view the labeled image from the example above

[http://127.0.0.1:8000/assets/a1b2c3d4e5_1678886400_predicted.jpg](http://127.0.0.1:8000/assets/a1b2c3d4e5_1678886400_predicted.jpg)

##### To view the labeled video

[http://127.0.0.1:8000/assets/f6g7h8i9j0_1678886500_predicted.mp4](http://127.0.0.1:8000/assets/f6g7h8i9j0_1678886500_predicted.mp4)

---

## Setup and Running

This project is designed to be run with Docker and Docker Compose.
Prerequisites

- Docker
- Docker Compose

1. Project Structure
   Ensure your project is structured as follows:

```
.
├── app/
│ ├── **init**.py
│ ├── config.py
│ └── main.py
│ ├── **init**.py
│ ├── config.py
│ └── main.py
├── assets/
│ └── models/
│ ├── <classifier>.pt
│ └── <detection>.pt
│ └── models/
│ ├── <classifier>.pt
│ └── <detection>.pt
├── builds/
│ ├── Dockerfile
│ └── entrypoint.sh
│ ├── Dockerfile
│ └── entrypoint.sh
├── .env
├── .env.example
├── compose.yaml
└── requirements.txt
```

2. Configuration

Copy the example environment file to create your local configuration:

`cp example.env .env`

You can now edit the .env file to change the server port, model paths, or supported file types. The default values will work out-of-the-box. 3. Add Your Models
You can now edit the .env file to change the server port, model paths, or supported file types. The default values will work out-of-the-box. 3. Add Your Models

Place your trained YOLO models into the assets/models/ directory. Ensure they are named cnn.pt (for classification) and detect.pt (for detection), or update the paths in your .env file. 4. Build and Run the Service
Place your trained YOLO models into the assets/models/ directory. Ensure they are named cnn.pt (for classification) and detect.pt (for detection), or update the paths in your .env file. 4. Build and Run the Service

From the root directory of the project, run the following command:

`docker-compose -f compose.yaml up --build`

--build: This flag is only needed the first time you run the service or after making changes to the Dockerfile, entrypoint.sh, or requirements.txt.

For subsequent runs, you can simply use docker-compose -f compose.yaml up.
The API will be available at http://127.0.0.1:8000 (or whichever port you configured in .env). 5. Stop the Service

To stop the running containers, press Ctrl+C in the terminal where compose is running, and then execute:

`docker-compose -f compose.yaml down`
