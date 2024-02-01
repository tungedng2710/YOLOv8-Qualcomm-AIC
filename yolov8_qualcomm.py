import numpy as np
import cv2
import qaic
from utils import xywh2xyxy, nms, draw_detections
import onnxruntime

class YOLOv8_Qualcomm():
    def __init__(self, 
                 model_path: str = 'weights/vehicle_640_bs1_c14/programqpc.bin',
                 aic_num_cores: int = 14,
                 num_activations: int = 1,
                 input_width: int = 640,
                 input_height: int = 640,
                 conf_thres: float = 0.5,
                 iou_thres: float = 0.5,
                 mode: str = 'qualcomm'):
        """
        Initialize YOLOv8 Detector
        Args:
            - model_path (str): path to model weight
            - aic_num_cores (int): number of AIC cores used for inference
            - num_activations (int): number of AIC activations
            - mode (str): onnx or qualcomm
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.mode = mode
        self.input_width = input_width
        self.input_height = input_height
        if mode == 'qualcomm':
            self.session = qaic.Session(model_path, aic_num_cores=aic_num_cores, num_activations=num_activations)
        elif mode == 'onnx':
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            model_inputs = self.session.get_inputs()
            self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
            model_outputs = self.session.get_outputs()
            self.output_names = [model_outputs[i].name for i in range(len(model_outputs))] 
        else:
            print(f"{mode} model is not supported")
            exit()

    def prepare_input(self, image):
        """
        Input must be a ndarray read by OpenCV
        """
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_img
    
    def infer(self, input_img):
        if self.mode == 'qualcomm':
            return self.session.run({"images": input_img})['output0']
        else:
            return self.session.run(self.output_names, {self.input_names[0]: input_img})
    
    def process_output(self, output):
        """
        Process inference's output to boxes
        """
        if self.mode == 'onnx':
            output = output[0]
        predictions = np.squeeze(output).T
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]
    
    def extract_boxes(self, predictions):
        """Extract boxes from predictions"""
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

        
if __name__ == '__main__':
    # mode = 'onnx'
    # model_path = 'weights/vehicle_yolov8s_640.onnx'
    mode = 'qualcomm'
    model_path = 'weights/vehicle_640_bs1_c14/programqpc.bin'
    detector = YOLOv8_Qualcomm(model_path=model_path, mode=mode)
    image = cv2.imread('data/anh1.png')
    input_image = detector.prepare_input(image)
    output = detector.infer(input_image)
    boxes, scores, class_ids = detector.process_output(output)
    result = draw_detections(image, boxes, scores, class_ids)
    cv2.imwrite(f'data/result_{mode}.jpg', result)
    
