from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class VisionConfig:
    motion_threshold: int = 2000
    motion_diff_threshold: int = 45
    motion_intensity_shift_threshold: float = 20.0
    settlement_seconds: float = 1.0
    processing_fps: float = 30.0
    contour_min_area: int = 650
    contour_max_area: int = 28100
    symbol_threshold_value: int = 211
    blank_pixel_ratio_threshold: float = 0.03
    line_peak_threshold: float = 0.38
    minimum_confidence: float = 0.55
    roi: Optional[tuple[int, int, int, int]] = None


@dataclass
class DetectionCandidate:
    box: tuple[int, int, int, int]
    sources: set[str]
    score: float


class VisionProcessor:
    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self.previous_gray: np.ndarray | None = None
        self.low_motion_frames = 0
        self.low_motion_started_at: float | None = None
        self.settlement_progress_seconds = 0.0

    @property
    def required_settlement_frames(self) -> int:
        return max(1, int(self.config.processing_fps * self.config.settlement_seconds))

    def reset_motion_history(self) -> None:
        self.previous_gray = None
        self.low_motion_frames = 0
        self.low_motion_started_at = None
        self.settlement_progress_seconds = 0.0

    def _resolve_roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame.shape[:2]
        if self.config.roi is None:
            return 0, 0, frame_w, frame_h

        x, y, w, h = self.config.roi
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        return x, y, w, h

    def detect_motion(self, frame: np.ndarray) -> int:
        x, y, w, h = self._resolve_roi(frame)
        roi_frame = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return self.config.motion_threshold + 1

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

    def update_settlement(self, motion_pixels: int, now_seconds: float | None = None) -> bool:
        if now_seconds is None:
            now_seconds = cv2.getTickCount() / cv2.getTickFrequency()

        motion_limit = self.config.motion_threshold
        grace_limit = int(motion_limit * 1.35)
        if motion_pixels < self.config.motion_threshold:
            self.low_motion_frames += 1
            if self.low_motion_started_at is None:
                self.low_motion_started_at = now_seconds
        elif motion_pixels < grace_limit and self.low_motion_started_at is not None:
            # Camera noise and exposure shifts can cause one-frame spikes after
            # the dice have stopped. Keep timing unless motion clearly resumes.
            self.low_motion_frames = max(0, self.low_motion_frames - 1)
        else:
            self.low_motion_frames = 0
            self.low_motion_started_at = None

        self.settlement_progress_seconds = (
            max(0.0, now_seconds - self.low_motion_started_at)
            if self.low_motion_started_at is not None
            else 0.0
        )

        return self.settlement_progress_seconds >= self.config.settlement_seconds

    @staticmethod
    def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        if right <= left or bottom <= top:
            return 0.0
        intersection = float((right - left) * (bottom - top))
        union = float((aw * ah) + (bw * bh)) - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        center_x = ix + (iw / 2.0)
        center_y = iy + (ih / 2.0)
        return ox <= center_x <= ox + ow and oy <= center_y <= oy + oh

    @staticmethod
    def _overlap_ratio_of_smaller(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        if right <= left or bottom <= top:
            return 0.0
        intersection = float((right - left) * (bottom - top))
        smaller = float(min(aw * ah, bw * bh))
        return intersection / smaller if smaller > 0 else 0.0

    def _dedupe_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for candidate in sorted(boxes, key=lambda box: box[2] * box[3], reverse=True):
            duplicate = False
            for existing in kept:
                if (
                    self._box_iou(candidate, existing) > 0.25
                    or self._overlap_ratio_of_smaller(candidate, existing) > 0.20
                    or self._center_inside(candidate, existing)
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        kept.sort(key=lambda box: (box[1], box[0]))
        return kept

    def _boxes_related(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        center_ax = ax + (aw / 2.0)
        center_ay = ay + (ah / 2.0)
        center_bx = bx + (bw / 2.0)
        center_by = by + (bh / 2.0)
        center_distance = ((center_ax - center_bx) ** 2 + (center_ay - center_by) ** 2) ** 0.5
        largest_side = float(max(aw, ah, bw, bh))
        return (
            self._box_iou(a, b) > 0.10
            or self._overlap_ratio_of_smaller(a, b) > 0.18
            or center_distance < largest_side * 0.60
        )

    @staticmethod
    def _source_weight(source: str) -> float:
        weights = {
            "body": 0.55,
            "square": 0.45,
            "symbol": 0.40,
        }
        return weights.get(source, 0.0)

    def _score_global_box(self, frame: np.ndarray, box: tuple[int, int, int, int], sources: set[str]) -> float:
        x, y, w, h = box
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return 0.0

        area = w * h
        area_score = 0.0
        if self.config.contour_min_area <= area <= self.config.contour_max_area:
            area_score = 0.20

        aspect_ratio = w / float(h)
        square_score = max(0.0, 0.20 - (abs(1.0 - aspect_ratio) * 0.16))
        signal_score = min(0.35, self._die_signal_score(frame, (x, y, w, h)) * 0.70)
        source_score = min(0.80, sum(self._source_weight(source) for source in sources))
        return source_score + signal_score + square_score + area_score

    def _combine_detection_signals(
        self,
        frame: np.ndarray,
        signal_boxes: list[tuple[str, tuple[int, int, int, int]]],
    ) -> list[tuple[int, int, int, int]]:
        clusters: list[list[tuple[str, tuple[int, int, int, int]]]] = []
        for source, box in signal_boxes:
            matched_cluster: list[tuple[str, tuple[int, int, int, int]]] | None = None
            for cluster in clusters:
                if any(self._boxes_related(box, existing_box) for _existing_source, existing_box in cluster):
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                clusters.append([(source, box)])
            else:
                matched_cluster.append((source, box))

        candidates: list[DetectionCandidate] = []
        for cluster in clusters:
            sources = {source for source, _box in cluster}
            best_box = max(
                (box for _source, box in cluster),
                key=lambda box: self._score_global_box(frame, box, sources),
            )
            score = self._score_global_box(frame, best_box, sources)
            if score >= 0.75:
                candidates.append(DetectionCandidate(box=best_box, sources=sources, score=score))

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return self._dedupe_boxes([candidate.box for candidate in candidates])

    @staticmethod
    def _top_face_box(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        # The camera sees dice sides. The symbol is on the top face, which is
        # the upper square-ish portion of the visible body.
        side = max(1, min(w, h))
        if w > h:
            x = x + int((w - side) / 2)
        return x, y, side, side

    @staticmethod
    def _die_signal_score(roi_frame: np.ndarray, box: tuple[int, int, int, int]) -> float:
        x, y, w, h = box
        crop = roi_frame[y : y + h, x : x + w]
        if crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        _hue, saturation, value = cv2.split(hsv)
        colored_ratio = float(cv2.countNonZero(cv2.inRange(saturation, 95, 255))) / float(w * h)
        bright_ratio = float(cv2.countNonZero(cv2.inRange(value, 190, 255))) / float(w * h)
        dark_colored_ratio = float(
            cv2.countNonZero(cv2.bitwise_and(cv2.inRange(value, 0, 105), cv2.inRange(saturation, 70, 255)))
        ) / float(w * h)
        return max(colored_ratio, bright_ratio * 0.85, dark_colored_ratio)

    @staticmethod
    def _touches_roi_boundary(box: tuple[int, int, int, int], roi_frame: np.ndarray, margin: int = 8) -> bool:
        x, y, w, h = box
        roi_h, roi_w = roi_frame.shape[:2]
        return x <= margin or y <= margin or x + w >= roi_w - margin or y + h >= roi_h - margin

    def _find_die_body_boxes(self, roi_frame: np.ndarray, roi_x: int, roi_y: int) -> list[tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        _hue, saturation, value = cv2.split(hsv)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        tray_saturation = float(np.median(saturation))
        tray_value = float(np.median(value))
        tray_gray = float(np.median(gray))

        # Segment dice as foreground objects against the tan tray. This supports
        # mixed dice colors: saturated colored dice, bright white dice, and dark
        # dice that contrast strongly with the tray.
        saturated = cv2.inRange(saturation, int(min(255, max(110, tray_saturation + 45))), 255)
        bright = cv2.inRange(value, int(min(255, tray_value + 55)), 255)
        dark = cv2.inRange(value, 0, int(max(0, tray_value - 45)))
        gray_delta = cv2.absdiff(gray, np.full_like(gray, int(tray_gray)))
        contrast = cv2.inRange(gray_delta, 35, 255)

        colored_dice = cv2.bitwise_and(saturated, cv2.inRange(value, 35, 255))
        bright_dice = cv2.bitwise_and(bright, contrast)
        dark_dice = cv2.bitwise_and(dark, saturated)
        body_mask = cv2.bitwise_or(colored_dice, cv2.bitwise_or(bright_dice, dark_dice))

        kernel = np.ones((5, 5), np.uint8)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            original_area = w * h
            contour_area = cv2.contourArea(contour)
            face_x, face_y, face_w, face_h = self._top_face_box(x, y, w, h)
            face_area = face_w * face_h
            if face_area < self.config.contour_min_area or face_area > self.config.contour_max_area:
                continue

            aspect_ratio = w / float(h)
            if not 0.35 <= aspect_ratio <= 2.40:
                continue

            extent = contour_area / float(original_area) if original_area > 0 else 0.0
            if extent < 0.30:
                continue

            face_box = (face_x, face_y, face_w, face_h)
            if self._touches_roi_boundary(face_box, roi_frame):
                continue
            if self._die_signal_score(roi_frame, face_box) < 0.08:
                continue

            boxes.append((face_x + roi_x, face_y + roi_y, face_w, face_h))

        return boxes

    def _find_symbol_hint_boxes(self, roi_frame: np.ndarray, roi_x: int, roi_y: int) -> list[tuple[int, int, int, int]]:
        roi_h, roi_w = roi_frame.shape[:2]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hint_boxes: list[tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            symbol_area = w * h
            if symbol_area < 350 or symbol_area > 18000:
                continue

            aspect_ratio = w / float(h)
            if not 0.25 <= aspect_ratio <= 4.0:
                continue

            side = int(max(w, h) * 3.2)
            side = max(70, min(side, 260))
            if side * side < self.config.contour_min_area or side * side > self.config.contour_max_area:
                continue

            center_x = x + (w / 2.0)
            center_y = y + (h / 2.0)
            face_x = int(max(0, min(center_x - (side / 2.0), roi_w - side)))
            face_y = int(max(0, min(center_y - (side / 2.0), roi_h - side)))
            face_box = (face_x, face_y, side, side)
            if self._touches_roi_boundary(face_box, roi_frame, margin=1):
                continue
            if self._die_signal_score(roi_frame, face_box) < 0.035:
                continue

            hint_boxes.append((face_x + roi_x, face_y + roi_y, side, side))

        return hint_boxes

    def _find_square_edge_boxes(self, roi_frame: np.ndarray, roi_x: int, roi_y: int) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 35, 125)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
            if not 4 <= len(approx) <= 8:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            face_x, face_y, face_w, face_h = self._top_face_box(x, y, w, h)
            face_box = (face_x, face_y, face_w, face_h)
            face_area = face_w * face_h
            if face_area < self.config.contour_min_area or face_area > self.config.contour_max_area:
                continue

            aspect_ratio = w / float(h)
            if not 0.45 <= aspect_ratio <= 2.20:
                continue

            contour_area = cv2.contourArea(contour)
            extent = contour_area / float(w * h) if w * h > 0 else 0.0
            if extent < 0.18:
                continue

            if self._touches_roi_boundary(face_box, roi_frame):
                continue
            if self._die_signal_score(roi_frame, face_box) < 0.05:
                continue

            boxes.append((face_x + roi_x, face_y + roi_y, face_w, face_h))

        return boxes

    def find_dice_contours(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        roi_x, roi_y, roi_w, roi_h = self._resolve_roi(frame)
        roi_frame = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        signal_boxes: list[tuple[str, tuple[int, int, int, int]]] = []
        signal_boxes.extend(("body", box) for box in self._find_die_body_boxes(roi_frame, roi_x, roi_y))
        signal_boxes.extend(("square", box) for box in self._find_square_edge_boxes(roi_frame, roi_x, roi_y))
        signal_boxes.extend(("symbol", box) for box in self._find_symbol_hint_boxes(roi_frame, roi_x, roi_y))
        return self._combine_detection_signals(frame, signal_boxes)

    def classify_symbol(self, die_crop: np.ndarray) -> tuple[str, float]:
        crop_h, crop_w = die_crop.shape[:2]
        margin_x = max(2, int(crop_w * 0.22))
        margin_y = max(2, int(crop_h * 0.22))
        face_crop = die_crop[margin_y : crop_h - margin_y, margin_x : crop_w - margin_x]
        if face_crop.size == 0:
            face_crop = die_crop

        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value_channel = cv2.split(hsv)
        median_value = float(np.median(value_channel))

        # Most tested Fate dice use painted light/yellow marks on saturated
        # colored bodies. A fixed grayscale threshold sees the body as ink, so
        # detect paint by relative brightness and paint-like color first.
        median_saturation = float(np.median(saturation))
        bright_offset = -5 if median_saturation > 80 else 30
        bright_floor = int(min(255, max(135, median_value + bright_offset)))
        bright_pixels = cv2.inRange(value_channel, bright_floor, 255)
        low_saturation_pixels = cv2.inRange(saturation, 0, 110)
        yellow_pixels = cv2.inRange(hue, 15, 45)
        binary_inv = cv2.bitwise_and(bright_pixels, cv2.bitwise_or(low_saturation_pixels, yellow_pixels))

        kernel = np.ones((3, 3), np.uint8)
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

        h, w = binary_inv.shape
        min_component_area = max(4, int(h * w * 0.003))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv)
        cleaned = np.zeros_like(binary_inv)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            center_x, center_y = centroids[label_id]
            component_w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            component_h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            near_face = (w * 0.05) <= center_x <= (w * 0.95) and (h * 0.05) <= center_y <= (h * 0.95)
            not_glare_blob = area <= int(h * w * 0.35)
            line_like = component_w >= int(w * 0.08) or component_h >= int(h * 0.08)
            if area >= min_component_area and near_face and not_glare_blob and line_like:
                cleaned[labels == label_id] = 255
        binary_inv = cleaned

        white_ratio = float(cv2.countNonZero(binary_inv)) / float(h * w)

        row_sums = np.sum(binary_inv > 0, axis=1)
        col_sums = np.sum(binary_inv > 0, axis=0)

        horizontal_peak = float(np.max(row_sums)) / float(w)
        vertical_peak = float(np.max(col_sums)) / float(h)
        horizontal_energy = float(np.mean(row_sums)) / float(w)
        vertical_energy = float(np.mean(col_sums)) / float(h)
        center_row_start = max(0, int(h * 0.35))
        center_row_end = min(h, int(h * 0.65))
        center_col_start = max(0, int(w * 0.35))
        center_col_end = min(w, int(w * 0.65))
        center_horizontal_peak = float(np.max(row_sums[center_row_start:center_row_end])) / float(w)
        center_vertical_peak = float(np.max(col_sums[center_col_start:center_col_end])) / float(h)

        if white_ratio >= self.config.blank_pixel_ratio_threshold and (
            center_horizontal_peak > self.config.line_peak_threshold
            and center_vertical_peak > 0.24
            and vertical_energy > 0.030
        ):
            confidence = min(
                1.0,
                0.55 + (center_horizontal_peak * 0.25) + (center_vertical_peak * 0.25) + vertical_energy,
            )
            confidence = max(self.config.minimum_confidence, confidence)
            return "+", confidence

        if (
            white_ratio >= self.config.blank_pixel_ratio_threshold * 0.65
            and center_horizontal_peak >= 0.15
            and center_horizontal_peak >= (center_vertical_peak * 0.65)
        ):
            confidence = min(1.0, 0.55 + (center_horizontal_peak * 0.90) + (white_ratio * 1.50))
            confidence = max(self.config.minimum_confidence, confidence)
            return "-", confidence

        if (
            white_ratio >= self.config.blank_pixel_ratio_threshold * 0.25
            and center_horizontal_peak >= 0.12
            and center_horizontal_peak >= (center_vertical_peak * 1.5)
        ):
            confidence = min(1.0, 0.55 + (center_horizontal_peak * 1.15) + (white_ratio * 2.00))
            confidence = max(self.config.minimum_confidence, confidence)
            return "-", confidence

        # Generic fallback for dark painted symbols. It is intentionally stricter
        # than the bright-paint path so blank colored dice are not read as marks.
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        dark_floor = int(max(0, min(self.config.symbol_threshold_value, median_value - 45)))
        _, dark_binary = cv2.threshold(gray, dark_floor, 255, cv2.THRESH_BINARY_INV)
        dark_binary = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, kernel, iterations=1)

        dark_num_labels, dark_labels, dark_stats, dark_centroids = cv2.connectedComponentsWithStats(dark_binary)
        dark_cleaned = np.zeros_like(dark_binary)
        for label_id in range(1, dark_num_labels):
            area = int(dark_stats[label_id, cv2.CC_STAT_AREA])
            center_x, center_y = dark_centroids[label_id]
            component_w = int(dark_stats[label_id, cv2.CC_STAT_WIDTH])
            component_h = int(dark_stats[label_id, cv2.CC_STAT_HEIGHT])
            near_center = (w * 0.15) <= center_x <= (w * 0.85) and (h * 0.15) <= center_y <= (h * 0.85)
            not_shadow_blob = area <= int(h * w * 0.35)
            line_like = component_w >= int(w * 0.14) or component_h >= int(h * 0.14)
            if area >= min_component_area and near_center and not_shadow_blob and line_like:
                dark_cleaned[dark_labels == label_id] = 255

        dark_ratio = float(cv2.countNonZero(dark_cleaned)) / float(h * w)
        if dark_ratio >= self.config.blank_pixel_ratio_threshold * 0.65:
            dark_rows = np.sum(dark_cleaned > 0, axis=1)
            dark_cols = np.sum(dark_cleaned > 0, axis=0)
            dark_horizontal_peak = float(np.max(dark_rows)) / float(w)
            dark_vertical_peak = float(np.max(dark_cols)) / float(h)
            dark_vertical_energy = float(np.mean(dark_cols)) / float(h)
            if (
                dark_horizontal_peak > self.config.line_peak_threshold
                and dark_vertical_peak > 0.30
                and dark_vertical_energy > 0.030
            ):
                confidence = min(
                    1.0,
                    0.55 + (dark_horizontal_peak * 0.25) + (dark_vertical_peak * 0.25) + dark_vertical_energy,
                )
                return "+", max(self.config.minimum_confidence, confidence)
            confidence = min(1.0, 0.55 + (dark_horizontal_peak * 0.90) + (dark_ratio * 1.50))
            return "-", max(self.config.minimum_confidence, confidence)

        confidence = max(
            self.config.minimum_confidence,
            1.0 - (white_ratio / self.config.blank_pixel_ratio_threshold),
        )
        if median_saturation >= 90 and white_ratio < self.config.blank_pixel_ratio_threshold * 0.65:
            # A clean saturated face is easier to separate from the tray than a
            # gray/white face. Treat that as positive evidence for a blank die,
            # but do not boost if paint-like pixels are close to symbol levels.
            color_blank_confidence = 0.70 + min(0.20, (median_saturation - 90.0) / 650.0)
            confidence = max(confidence, color_blank_confidence)
        return "blank", min(1.0, confidence)

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

    def preprocess_still_image(self, frame: np.ndarray) -> np.ndarray:
        # Keep color data because symbol classification uses paint color and
        # brightness relative to the die body.
        return frame

    def calculate_roll_from_still_image(self, frame: np.ndarray) -> list[dict]:
        preprocessed_frame = self.preprocess_still_image(frame)
        return self.calculate_roll(preprocessed_frame)

    def motion_mask_from_pair(self, previous_frame: np.ndarray, current_frame: np.ndarray) -> np.ndarray:
        prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        prev = cv2.GaussianBlur(prev, (5, 5), 0)
        curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        curr = cv2.GaussianBlur(curr, (5, 5), 0)
        delta = cv2.absdiff(prev, curr)
        _, thresh = cv2.threshold(delta, self.config.motion_diff_threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def edges_view(self, frame: np.ndarray) -> np.ndarray:
        roi_x, roi_y, roi_w, roi_h = self._resolve_roi(frame)
        roi_frame = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def contour_debug_view(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        for x, y, w, h in self.find_dice_contours(frame):
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(
                display,
                f"area {w * h}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        return display

    def thresholded_view(self, frame: np.ndarray) -> np.ndarray:
        roi_x, roi_y, roi_w, roi_h = self._resolve_roi(frame)
        roi_frame = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_inv = cv2.threshold(gray, self.config.symbol_threshold_value, 255, cv2.THRESH_BINARY_INV)
        return binary_inv
