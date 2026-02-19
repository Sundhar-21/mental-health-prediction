import cv2
import os
import shutil
import urllib.request

def setup_cascades():
    cascades_path = cv2.data.haarcascades
    print(f"OpenCV Cascades Path: {cascades_path}")

    files_to_get = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml"
    ]

    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"

    for file_name in files_to_get:
        # Check if already exists locally
        if os.path.exists(file_name):
            print(f"{file_name} already exists.")
            continue

        # Try to copy from OpenCV
        source_path = os.path.join(cascades_path, file_name)
        if os.path.exists(source_path):
            print(f"Copying {file_name} from OpenCV data...")
            shutil.copy(source_path, file_name)
        else:
            print(f"Downloading {file_name} from GitHub...")
            try:
                url = base_url + file_name
                urllib.request.urlretrieve(url, file_name)
                print(f"Downloaded {file_name}")
            except Exception as e:
                print(f"Failed to download {file_name}: {e}")

if __name__ == "__main__":
    setup_cascades()
