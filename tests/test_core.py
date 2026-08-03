import json
import socket
import struct
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from advanced_vision import AdvancedVisionEngine
from project_model import ProductRepository
from tcp_bridge import TcpVisionBridge


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(path)
    encoded.tofile(str(path))


class VisionCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = ProductRepository(Path(self.temp.name) / "products")
        self.product = self.repo.create("test")

    def tearDown(self):
        self.temp.cleanup()

    def _synthetic_scene(self):
        image = np.full((600, 800, 3), (50, 110, 45), np.uint8)
        bottom = cv2.boxPoints(((400, 430), (430, 115), -8)).astype(np.int32)
        top = cv2.boxPoints(((410, 175), (390, 105), 13)).astype(np.int32)
        cv2.fillConvexPoly(image, bottom, (30, 35, 95))
        cv2.fillConvexPoly(image, top, (125, 145, 150))
        return image, {"bottom": (400, 430), "top": (410, 175)}

    def test_learn_and_detect_two_pieces(self):
        image, samples = self._synthetic_scene()
        engine = AdvancedVisionEngine(self.product, lambda name: self.repo.load_image(self.product, name))
        for key, point in samples.items():
            piece = self.product.piece(key)
            piece.params.update({"method": "color", "tol_l": 8, "tol_ab": 8, "min_area_percent": 2})
            engine.sample_color(image, *point, piece)
            sprite, mask, _ = engine.learn_piece(image, piece)
            piece.template_file = self.repo.save_image(self.product, f"{key}_template.png", sprite)
            piece.mask_file = self.repo.save_image(self.product, f"{key}_mask.png", mask)
        detections, overlay, debug = engine.process(image, "deform")
        self.assertEqual({item.type_id for item in detections}, {0, 1})
        self.assertEqual(overlay.shape, image.shape)
        self.assertLess(debug.elapsed_ms, 1000)

    def test_product_roundtrip_and_clone(self):
        self.product.station.mm_per_pixel = 0.18
        self.product.piece("top").tcp_offset_local = [-20, -10]
        self.repo.save(self.product)
        loaded = self.repo.load("test")
        clone = self.repo.create("copy", loaded)
        self.assertEqual(clone.station.mm_per_pixel, 0.18)
        self.assertEqual(clone.piece("top").tcp_offset_local, [-20, -10])

    def test_homography_payload_roundtrip(self):
        self.product.station.homography = [1, 0, 10, 0, 1, 20, 0, 0, 1]
        self.repo.save(self.product)
        self.assertEqual(self.repo.load("test").station.homography[2:6], [10, 0, 1, 20])

    def test_automatic_brightness_threshold(self):
        image = np.full((240, 320, 3), 25, np.uint8)
        cv2.rectangle(image, (70, 80), (250, 170), (190, 190, 190), -1)
        piece = self.product.piece("top")
        piece.params.update({
            "method": "brightness",
            "auto_threshold": True,
            "invert": False,
            "min_area_percent": 3,
            "morph_iterations": 1,
        })
        engine = AdvancedVisionEngine(self.product, lambda _name: None)
        mask, contours = engine.segment_piece(image, piece)
        self.assertEqual(len(contours), 1)
        self.assertGreater(cv2.countNonZero(mask), 10000)

    def test_automatic_background_threshold(self):
        background = np.full((240, 320, 3), 50, np.uint8)
        scene = background.copy()
        cv2.rectangle(scene, (60, 70), (260, 180), (130, 160, 180), -1)
        self.product.background_file = self.repo.save_image(self.product, "background.png", background)
        piece = self.product.piece("top")
        piece.params.update({
            "method": "background",
            "auto_threshold": True,
            "min_area_percent": 3,
            "morph_iterations": 1,
        })
        engine = AdvancedVisionEngine(self.product, lambda name: self.repo.load_image(self.product, name))
        _, contours = engine.segment_piece(scene, piece)
        self.assertEqual(len(contours), 1)

    def test_merge_fragmented_piece(self):
        image = np.full((300, 500, 3), (20, 100, 20), np.uint8)
        cv2.rectangle(image, (70, 100), (210, 190), (40, 40, 150), -1)
        cv2.rectangle(image, (270, 100), (420, 190), (40, 40, 150), -1)
        piece = self.product.piece("top")
        piece.params.update({
            "method": "color",
            "merge_fragments": True,
            "min_area_percent": 1,
            "tol_l": 5,
            "tol_ab": 5,
        })
        engine = AdvancedVisionEngine(self.product, lambda _name: None)
        engine.sample_color(image, 100, 130, piece)
        _, contours = engine.segment_piece(image, piece)
        self.assertEqual(len(contours), 1)
        x, _, width, _ = cv2.boundingRect(contours[0])
        self.assertLessEqual(x, 75)
        self.assertGreaterEqual(width, 340)

    def test_clean_mask_keeps_largest_roi(self):
        mask = np.zeros((200, 300), np.uint8)
        cv2.rectangle(mask, (50, 60), (250, 150), 255, -1)
        cv2.rectangle(mask, (0, 0), (15, 15), 255, -1)
        clean = AdvancedVisionEngine.clean_mask(mask)
        self.assertEqual(clean[5, 5], 0)
        self.assertEqual(clean[100, 100], 255)

    def test_import_hoxco_product(self):
        source = Path(self.temp.name) / "hoxco"
        source.mkdir()
        for key, color in (("p1", (10, 30, 60, 255)), ("p2", (90, 110, 120, 255))):
            sprite = np.zeros((80, 180, 4), np.uint8)
            cv2.rectangle(sprite, (5, 5), (174, 74), color, -1)
            save_image(source / f"{key}.png", sprite)
        save_image(source / "bg.png", np.zeros((200, 300, 3), np.uint8))
        payload = {
            "top": "p2",
            "pose1": [0, 0, 0],
            "pose2": [12, -20, 4],
            "p1": {"tcp": [-4, -13], "med_lab": [21, 121, 127], "params": {"method": "color"}},
            "p2": {"tcp": [-20, -10], "med_lab": [88, 106, 136], "params": {"method": "color"}},
        }
        (source / "template.json").write_text(json.dumps(payload), encoding="utf-8")
        imported = self.repo.import_hoxco(source, "imported")
        self.assertEqual(imported.piece("top").tcp_offset_local, [-20, -10])
        self.assertEqual(imported.piece("top").design_x, 12)
        self.assertIsNotNone(self.repo.load_image(imported, imported.piece("top").mask_file))

    def test_tcp_bridge_protocol(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        image = np.zeros((60, 80, 3), np.uint8)
        encoded = cv2.imencode(".jpg", image)[1].tobytes()
        bridge = TcpVisionBridge(lambda frame: ([], frame))
        bridge.start("127.0.0.1", port)
        try:
            deadline = time.time() + 2
            while time.time() < deadline:
                try:
                    client = socket.create_connection(("127.0.0.1", port), timeout=.2)
                    break
                except OSError:
                    time.sleep(.02)
            else:
                self.fail("TCP bridge did not start")
            with client:
                client.sendall(struct.pack("<I", len(encoded)) + encoded)
                count = struct.unpack("<I", client.recv(4))[0]
                self.assertEqual(count, 0)
        finally:
            bridge.stop()


if __name__ == "__main__":
    unittest.main()
