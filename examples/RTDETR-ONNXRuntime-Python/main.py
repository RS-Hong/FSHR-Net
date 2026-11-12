# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import requests
import yaml


def download_file(url: str, local_path: str) -> str:
    """Download a file from a URL to a local path.

    Args:
        url (str): URL of the file to download.
        local_path (str): Local path where the file will be saved.
    """

    if os.path.exists(local_path):
        print(f"File already exists at {local_path}. Skipping download.")
        return local_path

    print(f"Downloading {url} to {local_path}...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return local_path


class RTDETR:
    """RT-DETR (Real-Time Detection Transformer) object detection model for ONNX inference and visualization.

    This class implements the RT-DETR model for object detection tasks, supporting ONNX model inference and
    visualization of detection results with bounding boxes and class labels.

    Attributes:
        model_path (str): Path to the ONNX model file.
        img_path (str): Path to the input image.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        session (ort.InferenceSession): ONNX runtime inference session.
        model_input (list): Model input metadata.
        input_width (int): Width dimension required by the model.
        input_height (int): Height dimension required by the model.
        classes (list[str]): List of class names from COCO dataset.
        color_palette (np.ndarray): Random color palette for visualization.
        img (np.ndarray): Loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.

    Methods:
        draw_detections: Draw bounding boxes and labels on the input image.
        preprocess: Preprocess the input image for model inference.
        bbox_cxcywh_to_xyxy: Convert bounding boxes from center format to corner format.
        postprocess: Postprocess model output to extract and visualize detections.
        main: Execute the complete object detection pipeline.

    Examples:
        Initialize RT-DETR detector and run inference
        >>> detector = RTDETR("rtdetr-l.onnx", "image.jpg", conf_thres=0.5)
        >>> output_image = detector.main()
        >>> cv2.imshow("Detections", output_image)
    """

    def __init__(
        self,
        model_path: str,
        img_path: str,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        class_names: str | None = None,
    ):
        """Initialize the RT-DETR object detection model.

        Args:
            model_path (str): Path to the ONNX model file.
            img_path (str): Path to the input image.
            conf_thres (float, optional): Confidence threshold for filtering detections.
            iou_thres (float, optional): IoU threshold for non-maximum suppression.
            class_names (Optional[str], optional): Path to a YAML file containing class names. If None, uses COCO
                dataset classes.
        """
        self.model_path = model_path
        self.img_path = img_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = class_names


        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model_path, providers=providers or available)

        self.model_input = self.session.get_inputs()
        self.input_width = self.model_input[0].shape[2]
        self.input_height = self.model_input[0].shape[3]

        if self.classes is None:

            self.classes = download_file(
                "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco8.yaml",
                "coco8.yaml",
            )


        with open(self.classes) as f:
            class_data = yaml.safe_load(f)
            self.classes = list(class_data["names"].values())


        if not isinstance(self.classes, list):
            raise ValueError("Classes should be a list of class names.")


        self.color_palette: np.ndarray = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, box: np.ndarray, score: float, class_id: int) -> None:
        """Draw bounding box and label on the input image for a detected object."""

        x1, y1, x2, y2 = box


        color = self.color_palette[class_id]


        cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


        label = f"{self.classes[class_id]}: {score:.2f}"


        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)


        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10


        cv2.rectangle(
            self.img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )


        cv2.putText(
            self.img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    def preprocess(self) -> np.ndarray:
        """Preprocess the input image for model inference.

        Loads the image, converts color space from BGR to RGB, resizes to model input dimensions, and normalizes pixel
        values to [0, 1] range.

        Returns:
            (np.ndarray): Preprocessed image data with shape (1, 3, H, W) ready for inference.
        """

        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found or unreadable: '{self.img_path}'")


        self.img_height, self.img_width = self.img.shape[:2]


        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)


        img = cv2.resize(img, (self.input_width, self.input_height))


        image_data = np.array(img) / 255.0


        image_data = np.transpose(image_data, (2, 0, 1))


        image_data = image_data[None].astype(np.float32)

        return image_data

    def bbox_cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert bounding boxes from center format to corner format.

        Args:
            boxes (np.ndarray): Array of shape (N, 4) where each row represents a bounding box in (center_x, center_y,
                width, height) format.

        Returns:
            (np.ndarray): Array of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max) format.
        """

        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2


        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height


        return np.column_stack((x_min, y_min, x_max, y_max))

    def postprocess(self, model_output: list[np.ndarray]) -> np.ndarray:
        """Postprocess model output to extract and visualize detections.

        Applies confidence thresholding, converts bounding box format, scales coordinates to original image dimensions,
        and draws detection annotations.

        Args:
            model_output (list[np.ndarray]): Output tensors from the model inference.

        Returns:
            (np.ndarray): Annotated image with detection bounding boxes and labels.
        """

        outputs = np.squeeze(model_output[0])


        boxes = outputs[:, :4]
        scores = outputs[:, 4:]


        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)


        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]


        boxes = self.bbox_cxcywh_to_xyxy(boxes)


        boxes[:, 0::2] *= self.img_width
        boxes[:, 1::2] *= self.img_height


        xywh_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        indices = cv2.dnn.NMSBoxes(xywh_boxes, scores.tolist(), self.conf_thres, self.iou_thres)
        indices = indices.flatten().tolist() if len(indices) else []


        for i in indices:
            self.draw_detections(boxes[i], float(scores[i]), int(labels[i]))

        return self.img

    def main(self) -> np.ndarray:
        """Execute the complete object detection pipeline on the input image.

        Performs preprocessing, ONNX model inference, and postprocessing to generate annotated detection results.

        Returns:
            (np.ndarray): Output image with detection annotations including bounding boxes and class labels.
        """

        image_data = self.preprocess()


        model_output = self.session.run(None, {self.model_input[0].name: image_data})


        return self.postprocess(model_output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rtdetr-l.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--img", type=str, default="bus.jpg", help="Path to the input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for non-maximum suppression.")
    args = parser.parse_args()


    detection = RTDETR(args.model, args.img, args.conf_thres, args.iou_thres)


    output_image = detection.main()


    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
