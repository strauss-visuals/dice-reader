from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VisionConfig:
    motion_pixel_threshold: int = 1200
    motion_diff_threshold: int = 30
    motion_intensity_shift_threshold: float = 20.0
    settlement_seconds: float = 1.5
    processing_fps: float = 30.0
    die_min_area: int = 500
    die_max_area: int = 30000
    blank_pixel_ratio_threshold: float = 0.03
    line_peak_threshold: float = 0.38
    minimum_confidence: float = 0.55


class VisionProcessor:
    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self.previous_gray: np.ndarray | None = None
        self.low_motion_frames = 0

    @property
    def required_settlement_frames(self) -> int:
        return max(1, int(self.config.processing_fps * self.config.settlement_seconds))

    def reset_motion_history(self) -> None:
        self.previous_gray = None
        self.low_motion_frames = 0

    def detect_motion(self, frame: np.ndarray) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return self.config.motion_pixel_threshold + 1

        previous_mean = float(np.mean(self.previous_gray))
        current_mean = float(np.mean(gray))
        if abs(current_mean - previous_mean) > self.config.motion_intensity_shift_threshold:
            self.previous_gray = gray
            return 0

        delta = cv2.absdiff(self.previous_gray, gray)
        _, thresh = cv2.threshold(delta, self.config.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_pixels = int(cv2.countNonZero(thresh))

        self.previous_gray = gray
        return motion_pixels

    def update_settlement(self, motion_pixels: int) -> bool:
        if motion_pixels < self.config.motion_pixel_threshold:
            self.low_motion_frames += 1
        else:
            self.low_motion_frames = 0

        return self.low_motion_frames >= self.required_settlement_frames

    def find_dice_contours(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dice_boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.config.die_min_area or area > self.config.die_max_area:
                continue

            aspect_ratio = w / float(h)
            if not 0.65 <= aspect_ratio <= 1.45:
                continue

            dice_boxes.append((x, y, w, h))

        dice_boxes.sort(key=lambda box: (box[1], box[0]))
        return dice_boxes

    def classify_symbol(self, die_crop: np.ndarray) -> tuple[str, float]:
        gray = cv2.cvtColor(die_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

        h, w = binary_inv.shape
        white_ratio = float(cv2.countNonZero(binary_inv)) / float(h * w)

        if white_ratio < self.config.blank_pixel_ratio_threshold:
            confidence = max(
                self.config.minimum_confidence,
                1.0 - (white_ratio / self.config.blank_pixel_ratio_threshold),
            )
            return "blank", min(1.0, confidence)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_inv)
        if num_labels > 1:
            largest_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary_inv = np.where(labels == largest_component, 255, 0).astype(np.uint8)

        row_sums = np.sum(binary_inv > 0, axis=1)
        col_sums = np.sum(binary_inv > 0, axis=0)

        horizontal_peak = float(np.max(row_sums)) / float(w)
        vertical_peak = float(np.max(col_sums)) / float(h)
        horizontal_energy = float(np.mean(row_sums)) / float(w)
        vertical_energy = float(np.mean(col_sums)) / float(h)

        if (
            horizontal_peak > self.config.line_peak_threshold
            and vertical_peak > self.config.line_peak_threshold
            and vertical_energy > 0.12
        ):
            confidence = min(1.0, (horizontal_peak + vertical_peak + vertical_energy) / 3.0)
            return "+", confidence

        if horizontal_peak >= vertical_peak:
            confidence = min(1.0, (horizontal_peak + horizontal_energy) / 2.0)
        else:
            confidence = min(1.0, (vertical_peak + vertical_energy) / 2.0)
        confidence = max(self.config.minimum_confidence, confidence)
        return "-", confidence

    def calculate_roll(self, frame: np.ndarray) -> list[dict]:
        dice_boxes = self.find_dice_contours(frame)
        results: list[dict] = []

        for x, y, w, h in dice_boxes:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            value, confidence = self.classify_symbol(crop)
            results.append(
                {
                    "value": value,
                    "confidence": float(confidence),
                    "bbox": [int(x), int(y), int(w), int(h)],
                }
            )

        return results
