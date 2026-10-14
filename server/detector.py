"""检测器：封装 ultralytics 模型，提供全局单例的加载与获取。"""

from __future__ import annotations

import os
import time

import numpy as np

from project_config import weights_by_priority

from .config import CONF, IMGSZ, IOU
from .schemas import Detection

# 本机为纯 CPU 推理：ONNX Runtime 默认会拉满全部逻辑核。限制线程后单帧分布更稳、
# 不被其它进程抢占（实测 intra=8 中位数 ≈29.6ms，不限线程 ≈34ms）。可用环境变量覆盖，
# 设 YOLO_ORT_THREADS=0 即不限制。
_ORT_THREADS = int(os.environ.get("YOLO_ORT_THREADS", "8"))


def _limit_onnxruntime_threads(intra: int, inter: int = 1) -> None:
    """让 ultralytics 加载 .onnx 时使用受限线程数的 ONNX Runtime 会话。

    ultralytics 构造 ORT 会话时 session_options 默认为 None，会走 ORT 默认值（用满所有核）。
    这里在 YOLO 加载前替换 onnxruntime.InferenceSession，注入 intra/inter 线程数。
    """
    import onnxruntime

    original = onnxruntime.InferenceSession
    if getattr(original, "_yolo_thread_limited", False):
        return

    def patched(path_or_bytes, sess_options=None, **kwargs):
        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()
        try:
            sess_options.intra_op_num_threads = intra
            sess_options.inter_op_num_threads = inter
        except Exception:  # 某些 EP 的 options 不支持设置线程数，忽略即可
            pass
        return original(path_or_bytes, sess_options, **kwargs)

    patched._yolo_thread_limited = True
    onnxruntime.InferenceSession = patched


class Detector:
    """封装 ultralytics 模型，线程安全地执行推理。

    加载 .onnx 时，ultralytics 底层即使用 ONNX Runtime 作为推理后端。
    """

    def __init__(
        self, weights: str, imgsz: int = IMGSZ, conf: float = CONF, iou: float = IOU
    ) -> None:
        from ultralytics import YOLO

        if str(weights).endswith(".onnx"):
            _limit_onnxruntime_threads(_ORT_THREADS)
        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = self.model.names
        self.weights = weights

    def detect(self, bgr: np.ndarray) -> tuple[list[Detection], float]:
        """对一张 BGR 图像推理，返回检测框列表与耗时(ms)。"""
        t0 = time.perf_counter()
        result = self.model.predict(
            source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        dets: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append(
                    Detection(
                        cls=int(k),
                        label=str(self.names.get(int(k), k)),
                        conf=float(c),
                        xyxy=[float(x1), float(y1), float(x2), float(y2)],
                    )
                )
        return dets, infer_ms


_service: Detector | None = None


def load_detector() -> Detector:
    """加载（或复用）全局检测器实例，供服务启动时调用。"""
    global _service
    if _service is None:
        weights = weights_by_priority()
        print(f"[server] loading model: {weights}")
        _service = Detector(weights)
        print(f"[server] model ready, {len(_service.names)} classes")
    return _service


def get_detector() -> Detector:
    """获取已加载的检测器；未就绪时抛错，避免调用空实例。"""
    if _service is None:
        raise RuntimeError("detector 尚未加载")
    return _service


def peek_detector() -> Detector | None:
    """返回当前的检测器实例（可能为 None）。"""
    return _service
