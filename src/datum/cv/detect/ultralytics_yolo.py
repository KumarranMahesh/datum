"""Ultralytics YOLO detector adapter.

LICENSE NOTE. This adapter wraps the `ultralytics` package, which is
licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
Datum itself remains Apache-2.0; using this adapter does not change that.
However, if you deploy Datum with the `yolo` extra installed in a way that
exposes it over a network (a hosted service, for instance), AGPL's
network-use clause may apply to that deployment. If commercial deployment
is on your roadmap, a permissively licensed detector (RT-DETR via
transformers, torchvision Faster R-CNN) slots in behind the same
`Detector` interface and avoids the question entirely.

Install with `uv sync --extra yolo` or `pip install datum[yolo]`.
Model checkpoints (~6 MB for the default `yolo11n.pt`) are downloaded by
ultralytics on first use and cached under `~/.cache/ultralytics/`.
"""

from __future__ import annotations

from datum.cv.detect import (
    Detection,
    DetectionBatch,
    Detector,
    FrameBatch,
    register,
)

# Standard COCO class indices: 0 is person, 32 is sports ball. Filtering
# to these by default catches the two things broadcast football analytics
# almost always wants. Override via the detector_config in CvConfig.
_DEFAULT_CLASS_FILTER: list[int] = [0, 32]


@register("ultralytics-yolo")
class UltralyticsYoloDetector(Detector):
    """YOLO11 / YOLOv8 detection via the ultralytics package.

    Constructor parameters become the detector_config in CvConfig, so
    anything passed here ends up in the run id hash. Keep values JSON-
    compatible (no torch.device objects, no callables).
    """

    def __init__(
        self,
        *,
        model_name: str = "yolo11n.pt",
        device: str = "auto",
        class_filter: list[int] | None = None,
        imgsz: int = 640,
        min_confidence: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        # Heavy imports live here so the module can be imported (for
        # registration) without forcing ultralytics or torch to load on a
        # base install.
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run `uv sync --extra yolo` "
                "(or `pip install ultralytics`) to enable this detector."
            ) from e

        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = YOLO(model_name)
        self._device = device
        self._class_filter = (
            class_filter if class_filter is not None else _DEFAULT_CLASS_FILTER
        )
        self._imgsz = imgsz
        self._min_confidence = min_confidence
        self._iou = iou

        # ultralytics exposes the class names dict as `model.names`.
        # Filter to the allowed set so class_map only advertises what
        # this detector can actually emit.
        all_names = dict(self._model.names)
        if self._class_filter is not None:
            self._class_map: dict[int, str] = {
                i: all_names[i] for i in self._class_filter if i in all_names
            }
        else:
            self._class_map = all_names

    @property
    def class_map(self) -> dict[int, str]:
        return dict(self._class_map)

    def detect(self, frames: FrameBatch) -> DetectionBatch:
        if len(frames) == 0:
            return DetectionBatch(per_frame=[])

        # ultralytics' predict() accepts a list of HxWx3 ndarrays. The 4D
        # batch is broken apart here; ultralytics handles the rest.
        results = self._model.predict(
            source=list(frames),
            device=self._device,
            imgsz=self._imgsz,
            conf=self._min_confidence,
            iou=self._iou,
            classes=self._class_filter,
            verbose=False,
        )

        per_frame: list[list[Detection]] = []
        for result in results:
            frame_detections: list[Detection] = []
            if result.boxes is None or len(result.boxes) == 0:
                per_frame.append(frame_detections)
                continue

            # Tensors live on the model's device until pulled to CPU.
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)

            for box, c, k in zip(xyxy, conf, cls, strict=True):
                frame_detections.append(
                    Detection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(c),
                        class_id=int(k),
                    )
                )
            per_frame.append(frame_detections)

        return DetectionBatch(per_frame=per_frame)


__all__ = ["UltralyticsYoloDetector"]
