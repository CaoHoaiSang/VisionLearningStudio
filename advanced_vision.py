from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from project_model import PieceModel, ProductModel
from vision_core import Detection


ALGORITHMS = {
    "auto": "Tự động (khuyến nghị)",
    "features": "Đặc trưng vải — xoay 360°, che khuất",
    "gray": "So khớp mẫu — ảnh xám",
    "deform": "So khớp mẫu — biến dạng",
    "background": "Chỉ tách nền",
}


@dataclass
class MatchDebug:
    algorithm: str
    elapsed_ms: float
    notes: list[str]
    masks: dict[str, np.ndarray]


def normalize_angle(angle: float) -> float:
    return ((float(angle) + 90.0) % 180.0) - 90.0


def long_axis_rect(contour: np.ndarray):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0
    return (float(cx), float(cy)), (float(w), float(h)), normalize_angle(angle)


def rotate(offset, angle_deg):
    x, y = float(offset[0]), float(offset[1])
    r = math.radians(angle_deg)
    return x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)


class AdvancedVisionEngine:
    def __init__(self, product: ProductModel, image_loader: Callable[[str | None], np.ndarray | None]):
        self.product = product
        self.image_loader = image_loader
        self._lock = threading.RLock()
        self.last_debug = MatchDebug("auto", 0.0, [], {})

    def update_product(self, product: ProductModel, image_loader: Callable[[str | None], np.ndarray | None]):
        with self._lock:
            self.product = product
            self.image_loader = image_loader

    def segment_piece(
        self,
        image: np.ndarray,
        piece: PieceModel,
        method_override: str | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        params = piece.params
        method = method_override or params.get("method", "color")
        blur = max(1, int(params.get("blur", 9)))
        blur += 1 - blur % 2
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        if method == "background":
            background = self.image_loader(self.product.background_file)
            if background is None:
                return np.zeros(image.shape[:2], np.uint8), []
            if background.shape[:2] != image.shape[:2]:
                background = cv2.resize(background, (image.shape[1], image.shape[0]))
            diff = cv2.absdiff(image, background[:, :, :3])
            value = np.max(diff, axis=2)
            value = cv2.GaussianBlur(value, (blur, blur), 0)
            threshold = int(params.get("bg_threshold", params.get("threshold", 20)))
            flag = cv2.THRESH_BINARY
            if params.get("auto_threshold", False):
                threshold = 0
                flag |= cv2.THRESH_OTSU
            _, mask = cv2.threshold(value, threshold, 255, flag)
        elif method == "brightness":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (blur, blur), 0)
            flag = cv2.THRESH_BINARY_INV if params.get("invert", False) else cv2.THRESH_BINARY
            threshold = int(params.get("threshold", 130))
            if params.get("auto_threshold", False):
                flag |= cv2.THRESH_OTSU
                threshold = 0
            _, mask = cv2.threshold(gray, threshold, 255, flag)
        else:
            if piece.reference_lab is None:
                return np.zeros(image.shape[:2], np.uint8), []
            target = np.asarray(piece.reference_lab, np.float32)
            delta = lab.astype(np.float32) - target.reshape(1, 1, 3)
            tol_l = float(params.get("tol_l", params.get("color_tolerance", 15) * 1.8))
            tol_ab = float(params.get("tol_ab", params.get("color_tolerance", 15)))
            mask = np.where(
                (np.abs(delta[:, :, 0]) <= tol_l)
                & (np.sqrt(delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2) <= tol_ab),
                255,
                0,
            ).astype(np.uint8)

        kernel_size = max(1, int(params.get("morph_kernel", 7)))
        kernel_size += 1 - kernel_size % 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = max(0, int(params.get("morph_iterations", 2)))
        if iterations:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum = image.shape[0] * image.shape[1] * float(params.get("min_area_percent", 3)) / 100
        contours = [c for c in contours if cv2.contourArea(c) >= minimum]
        contours.sort(key=cv2.contourArea, reverse=True)
        if params.get("merge_fragments", False) and len(contours) > 1:
            merged = cv2.convexHull(np.vstack(contours))
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [merged], -1, 255, -1)
            contours = [merged]
        return mask, contours

    def learn_piece(self, image: np.ndarray, piece: PieceModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask, contours = self.segment_piece(image, piece)
        if not contours:
            raise ValueError(f"Không tìm thấy {piece.name}; hãy chỉnh tham số tách")
        contour = contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        pad = 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        sprite = image[y0:y1, x0:x1].copy()
        sprite_mask = mask[y0:y1, x0:x1].copy()
        sprite_mask = self.clean_mask(sprite_mask)
        contour_local = contour - np.array([[[x0, y0]]], dtype=contour.dtype)
        return sprite, sprite_mask, contour_local

    def sample_color(self, image: np.ndarray, u: int, v: int, piece: PieceModel, radius=8):
        h, w = image.shape[:2]
        u, v = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
        patch = image[max(0, v - radius):min(h, v + radius + 1), max(0, u - radius):min(w, u + radius + 1)]
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        piece.reference_lab = np.median(lab.reshape(-1, 3), axis=0).astype(float).tolist()
        return piece.reference_lab

    def process(self, image: np.ndarray, algorithm: str | None = None):
        with self._lock:
            t0 = time.perf_counter()
            algorithm = algorithm or self.product.station.algorithm
            notes, masks = [], {}
            detections: list[Detection] = []
            occupied = np.zeros(image.shape[:2], np.uint8)
            pieces = sorted(self.product.pieces, key=lambda p: not p.is_top)

            for piece in pieces:
                selected = algorithm
                if selected == "auto":
                    template = self.image_loader(piece.template_file)
                    minimum_features = int(piece.params.get("pattern_threshold", 30))
                    selected = "features" if self._feature_count(template) >= minimum_features else "deform"
                detection = None
                if selected == "features":
                    detection = self._match_features(image, piece)
                    if detection is None:
                        notes.append(f"{piece.name}: đặc trưng không đủ, dùng tách/contour")
                        detection, mask = self._match_contour(image, piece)
                        masks[piece.key] = mask
                elif selected == "gray":
                    detection = self._match_gray(image, piece)
                    if detection is None:
                        detection, mask = self._match_contour(image, piece)
                        masks[piece.key] = mask
                else:
                    forced_method = "background" if selected == "background" else None
                    detection, mask = self._match_contour(image, piece, forced_method)
                    masks[piece.key] = mask
                if detection is not None:
                    if cv2.countNonZero(cv2.bitwise_and(occupied, self._contour_mask(image.shape[:2], detection.contour))) == 0:
                        detections.append(detection)
                        cv2.drawContours(occupied, [detection.contour], -1, 255, -1)
                    else:
                        detections.append(detection)

            overlay = self.draw(image, detections)
            elapsed = (time.perf_counter() - t0) * 1000
            self.last_debug = MatchDebug(algorithm, elapsed, notes, masks)
            return detections, overlay, self.last_debug

    def _match_contour(self, image: np.ndarray, piece: PieceModel, method_override=None):
        mask, contours = self.segment_piece(image, piece, method_override)
        template_mask = self.image_loader(piece.mask_file)
        template_contour = None
        if template_mask is not None:
            if template_mask.ndim == 3:
                template_mask = cv2.cvtColor(template_mask, cv2.COLOR_BGR2GRAY)
            template_contours, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if template_contours:
                template_contour = max(template_contours, key=cv2.contourArea)
        best, best_score = None, float("inf")
        for contour in contours:
            score = cv2.matchShapes(template_contour, contour, cv2.CONTOURS_MATCH_I1, 0) if template_contour is not None else 0
            if score < best_score:
                best, best_score = contour, score
        if best is None:
            return None, mask
        return self._detection_from_contour(best, piece, max(0.0, 1.0 - best_score)), mask

    def _match_features(self, image: np.ndarray, piece: PieceModel):
        template = self.image_loader(piece.template_file)
        if template is None:
            return None
        template = template[:, :, :3]
        gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1600, fastThreshold=8)
        k1, d1 = orb.detectAndCompute(gray_t, None)
        k2, d2 = orb.detectAndCompute(gray_s, None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(d1, d2, k=2)
        good = [a for a, b in pairs if a.distance < 0.76 * b.distance]
        if len(good) < 8:
            return None
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if matrix is None:
            return None
        template_mask = self.image_loader(piece.mask_file)
        contour = None
        if template_mask is not None:
            if template_mask.ndim == 3:
                template_mask = cv2.cvtColor(template_mask, cv2.COLOR_BGR2GRAY)
            source_contours, _ = cv2.findContours(template_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if source_contours:
                source_contour = max(source_contours, key=cv2.contourArea).astype(np.float32)
                contour = cv2.perspectiveTransform(source_contour, matrix).astype(np.int32)
        if contour is None:
            h, w = gray_t.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            contour = cv2.perspectiveTransform(corners, matrix).astype(np.int32)
        score = float(inliers.sum() / max(1, len(inliers))) if inliers is not None else 0
        return self._detection_from_contour(contour, piece, score)

    def _match_gray(self, image: np.ndarray, piece: PieceModel):
        template = self.image_loader(piece.template_file)
        if template is None:
            return None
        scene = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
        scene_small = cv2.resize(scene, None, fx=0.5, fy=0.5)
        tolerance = max(0.05, self.product.station.scale_tolerance_percent / 100)
        best = (-1.0, None, None)
        for scale in np.linspace(1 - tolerance, 1 + tolerance, 3):
            for angle in range(-90, 91, 15):
                candidate = self._rotate_bound(template_gray, angle, scale * 0.5)
                if candidate.shape[0] >= scene_small.shape[0] or candidate.shape[1] >= scene_small.shape[1]:
                    continue
                edges_t = cv2.Canny(candidate, 40, 120)
                edges_s = cv2.Canny(scene_small, 40, 120)
                result = cv2.matchTemplate(edges_s, edges_t, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(result)
                if score > best[0]:
                    best = (score, location, (candidate.shape[1], candidate.shape[0], angle))
        if best[1] is None or best[0] < 0.2:
            return None
        x, y = best[1]
        w, h, angle = best[2]
        contour = cv2.boxPoints(((2 * x + w, 2 * y + h), (2 * w, 2 * h), angle)).astype(np.int32).reshape(-1, 1, 2)
        return self._detection_from_contour(contour, piece, best[0])

    def _detection_from_contour(self, contour, piece, score):
        (cx, cy), (w, h), angle = long_axis_rect(contour)
        dx, dy = rotate(piece.tcp_offset_local, angle)
        return Detection(
            type_id=piece.type_id,
            class_name=piece.name,
            center_u=cx,
            center_v=cy,
            tcp_u=cx + dx,
            tcp_v=cy + dy,
            width=w,
            height=h,
            angle=angle,
            area=float(abs(cv2.contourArea(contour))),
            score=float(score),
            contour=np.asarray(contour, np.int32),
        )

    def draw(self, image, detections):
        overlay = image.copy()
        for detection in detections:
            piece = next(p for p in self.product.pieces if p.type_id == detection.type_id)
            color = tuple(piece.draw_color_bgr)
            cv2.drawContours(overlay, [detection.contour], -1, color, 2, cv2.LINE_AA)
            center = (round(detection.tcp_u), round(detection.tcp_v))
            cv2.drawMarker(overlay, center, (0, 50, 255), cv2.MARKER_CROSS, 26, 2)
            dx, dy = rotate((80, 0), detection.angle)
            cv2.arrowedLine(overlay, center, (round(center[0] + dx), round(center[1] + dy)), color, 2, tipLength=.25)
            label = "MANH TREN (GAP)" if piece.is_top else "MANH DUOI"
            cv2.putText(overlay, f"{label}  {detection.angle:+.1f} deg  {detection.score:.2f}",
                        (max(4, center[0] - 90), max(22, center[1] - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
        return overlay

    @staticmethod
    def _feature_count(image):
        if image is None:
            return 0
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        return len(cv2.ORB_create(nfeatures=500, fastThreshold=8).detect(gray, None))

    @staticmethod
    def _rotate_bound(image, angle, scale=1.0):
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        matrix[0, 2] += nw / 2 - w / 2
        matrix[1, 2] += nh / 2 - h / 2
        return cv2.warpAffine(image, matrix, (nw, nh), borderValue=0)

    @staticmethod
    def _contour_mask(shape, contour):
        mask = np.zeros(shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        return mask

    @staticmethod
    def clean_mask(mask, kernel_size=5):
        if mask is None:
            return mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        kernel_size = max(3, int(kernel_size) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        clean = np.zeros_like(mask)
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        return clean
