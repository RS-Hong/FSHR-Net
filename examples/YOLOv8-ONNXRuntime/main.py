# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLOv8:
    """YOLOv8 object detection model class for handling ONNX inference and visualization.

    This class provides functionality to load a YOLOv8 ONNX model, perform inference on images, and visualize the
    detection results with bounding boxes and labels.

    Attributes:
        onnx_model (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        confidence_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        classes (list[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        letterbox: Resize and reshape images while maintaining aspect ratio by adding padding.
        draw_detections: Draw bounding boxes and labels on the input image based on detected objects.
        preprocess: Preprocess the input image before performing inference.
        postprocess: Perform post-processing on the model's output to extract and visualize detections.
        main: Perform inference using an ONNX model and return the output image with drawn detections.

    Examples:
        Initialize YOLOv8 detector and run inference
        >>> detector = YOLOv8("yolov8n.onnx", "image.jpg", 0.5, 0.5)
        >>> output_image = detector.main()
    """

    def __init__(self, onnx_model: str, input_image: str, confidence_thres: float, iou_thres: float):
        """Initialize an instance of the YOLOv8 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres


        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]


        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def letterbox(self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            img (np.ndarray): Resized and padded image.
            pad (tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]


        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])


        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: list[float], score: float, class_id: int) -> None:
        """Draw bounding boxes and labels on the input image based on the detected objects."""

        x1, y1, w, h = box


        color = self.color_palette[class_id]


        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)


        label = f"{self.classes[class_id]}: {score:.2f}"


        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)


        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10


        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )


        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            image_data (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            pad (tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """

        self.img = cv2.imread(self.input_image)


        self.img_height, self.img_width = self.img.shape[:2]


        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))


        image_data = np.array(img) / 255.0


        image_data = np.transpose(image_data, (2, 0, 1))


        image_data = image_data[None].astype(np.float32)


        return image_data, pad

    def postprocess(self, input_image: np.ndarray, output: list[np.ndarray], pad: tuple[int, int]) -> np.ndarray:
        """Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs. It applies
        non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (list[np.ndarray]): The output arrays from the model.
            pad (tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """

        outputs = np.transpose(np.squeeze(output[0]))


        rows = outputs.shape[0]


        boxes = []
        scores = []
        class_ids = []


        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]


        for i in range(rows):

            classes_scores = outputs[i][4:]


            max_score = np.amax(classes_scores)


            if max_score >= self.confidence_thres:

                class_id = np.argmax(classes_scores)


                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]


                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)


                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])


        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)


        for i in np.array(indices).flatten():

            box = boxes[int(i)]
            score = scores[int(i)]
            class_id = class_ids[int(i)]


            self.draw_detections(input_image, box, score, class_id)


        return input_image

    def main(self) -> np.ndarray:
        """Perform inference using an ONNX model and return the output image with drawn detections.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        session = ort.InferenceSession(self.onnx_model, providers=providers or available)


        model_inputs = session.get_inputs()


        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]


        img_data, pad = self.preprocess()


        outputs = session.run(None, {model_inputs[0].name: img_data})


        return self.postprocess(self.img, outputs, pad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()


    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")


    detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)


    output_image = detection.main()


    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)


    cv2.waitKey(0)
