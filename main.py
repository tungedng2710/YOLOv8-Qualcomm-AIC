import cv2
import time
from utils import draw_detections, is_image_file, is_video_file
from yolov8_qualcomm import YOLOv8_Qualcomm

SOURCE = 'data/video1.mp4'
# MODE = 'onnx'
# MODEL_PATH = 'weights/vehicle_yolov8s_640.onnx'
MODE = 'qualcomm'
MODEL_PATH = 'weights/vehicle_640_bs1_c14/programqpc.bin'

def predict(detector, image):
    """
    Run yolov8 detector and draw the result
    """
    input_image = detector.prepare_input(image)
    output = detector.infer(input_image)
    boxes, scores, class_ids = detector.process_output(output)
    result = draw_detections(image, boxes, scores, class_ids)
    return result


if __name__ == '__main__':
    detector = YOLOv8_Qualcomm(model_path=MODEL_PATH, mode=MODE)
    if is_image_file(SOURCE):
        image = cv2.imread(SOURCE)
        result = predict(detector, image)
        
    elif is_video_file(SOURCE):
        cap = cv2.VideoCapture(SOURCE)
        start_time = time.time()
        num_frames = 0
        if not cap.isOpened():
            print(f"Error: Could not open video file '{SOURCE}'")
            exit()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            num_frames += 1
            frame = predict(detector, frame)
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = num_frames / elapsed_time
            print(f"FPS: {int(fps)}")
            # cv2.imshow('Video', frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
    else:
        print("Something wrong. Exit now")