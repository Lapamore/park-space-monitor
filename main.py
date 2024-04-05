import numpy as np
import cv2
from ultralytics import YOLO
import threading


def load_model(path_to_model: str):
    return YOLO(path_to_model, verbose=False)


def proccess_frame(frame, model, label_colors):
    result = model(frame)
    data = result[0].obb
    xyxy = data.xyxy.cpu().numpy()
    cls = data.cls.cpu().numpy()
    conf = data.conf.cpu().numpy()

    for box, label, confidence in zip(xyxy, cls, conf):
        if confidence > 0.75:
            x, y, x1, y1 = list(map(int, box))
            cv2.rectangle(frame, (x, y), (x1, y1), label_colors[int(label)], 2)

    return frame


def display_video(cap, model, label_colors):
    while cap.isOpened():
        success, frame = cap.read()
        image_resize = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        if not success:
            break
        
        image_resize = proccess_frame(image_resize, model, label_colors)
        cv2.imshow("Window", image_resize)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


def main():
    PATH_TO_VIDEO = "YOUR_PATH_TO_VIDEO"
    PATH_TO_MODEL = "model/best.pt"
    LABEL_COLORS = [(0, 255, 0), (0, 0, 255)]

    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    model = load_model(PATH_TO_MODEL)

    display_threading = threading.Thread(target = display_video, args=(cap, model, LABEL_COLORS))
    display_threading.start()
    display_threading.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
