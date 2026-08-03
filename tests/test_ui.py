import time
import unittest
from types import SimpleNamespace

import customtkinter as ctk
import cv2

from advanced_vision import normalize_angle
from modern_app import VisionLab


class ModernUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = VisionLab()
        cls.app.withdraw()
        cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app._close()

    def test_b1_is_scrollable_and_sources_switch(self):
        self.assertIsInstance(self.app.b1_scroll, ctk.CTkScrollableFrame)
        canvas = self.app.b1_scroll._parent_canvas
        canvas.update_idletasks()
        self.assertGreater(canvas.bbox("all")[3], canvas.winfo_height())
        canvas.yview_moveto(1)
        canvas.update_idletasks()
        self.assertAlmostEqual(canvas.yview()[1], 1.0, places=2)
        self.app._source_mode_changed("Robot · DeltaX")
        self.assertTrue(self.app.robot_source.winfo_manager())
        self.assertFalse(self.app.static_source.winfo_manager())
        self.app._source_mode_changed("Ảnh tĩnh")
        self.assertTrue(self.app.static_source.winfo_manager())

    def test_b2_drag_is_direct_and_fast(self):
        self.app.layout_view.configure(width=1000, height=700)
        self.app.layout_view.update()
        self.app.layout_view.render()
        self.assertIn("top", self.app.layout_view.items)
        top_piece = self.app.product.piece("top")
        original = (top_piece.design_x, top_piece.design_y)
        self.app.layout_view.drag_key = "top"
        self.app.layout_view.drag_anchor = (100, 100)
        started = time.perf_counter()
        for index in range(200):
            self.app.layout_view._drag(SimpleNamespace(x=101 + index, y=100 + index // 3))
        elapsed = time.perf_counter() - started
        top_piece.design_x, top_piece.design_y = original
        self.app.layout_view.drag_key = None
        self.app.layout_view.drag_anchor = None
        self.assertLess(elapsed, 1.0)

    def test_radxa_sample_detects_two_pieces_and_clean_roi(self):
        self.app._detect()
        self.assertEqual({item.type_id for item in self.app.detections}, {0, 1})
        shown, _origin, mask = self.app._masked_roi("top")
        original = self.app._asset(self.app.product.piece("top").template_file)
        self.assertIsNotNone(mask)
        self.assertLessEqual(shown.shape[0], original.shape[0])
        self.assertLessEqual(shown.shape[1], original.shape[1])

    def test_ghost_angle_matches_b2_visual_relation(self):
        self.app._detect()
        bottom_detection = next(item for item in self.app.detections if item.type_id == 0)
        bottom = self.app.product.piece("bottom")
        top = self.app.product.piece("top")
        bottom_reference = self.app._mask_axis_angle(bottom)
        top_reference = self.app._mask_axis_angle(top)
        _u, _v, sprite_rotation = self.app._ghost_transform(bottom_detection)
        expected_relative = (
            (top_reference + top.design_angle)
            - (bottom_reference + bottom.design_angle)
        )
        actual_relative = (
            (top_reference + sprite_rotation)
            - bottom_detection.angle
        )
        self.assertAlmostEqual(
            normalize_angle(actual_relative),
            normalize_angle(expected_relative),
            places=5,
        )

    def test_tcp_can_be_set_to_mask_centroid(self):
        piece = self.app.product.piece("top")
        original = list(piece.tcp_offset_local)
        mask = self.app._asset(piece.mask_file)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = self.app.engine.clean_mask(mask)
        moments = cv2.moments(mask, binaryImage=True)
        expected = [
            moments["m10"] / moments["m00"] - mask.shape[1] / 2,
            moments["m01"] / moments["m00"] - mask.shape[0] / 2,
        ]
        self.app._tcp_to_object_center("top")
        self.assertAlmostEqual(piece.tcp_offset_local[0], expected[0], places=5)
        self.assertAlmostEqual(piece.tcp_offset_local[1], expected[1], places=5)
        piece.tcp_offset_local = original
        self.app._tcp_render()

    def test_all_learning_labs_render(self):
        for index in range(15):
            self.app.lessons.selection_clear(0, "end")
            self.app.lessons.selection_set(index)
            self.app._lesson(None)
            if self.app._lesson_job:
                self.app.after_cancel(self.app._lesson_job)
                self.app._lesson_job = None
            self.app._render_current_lesson()
            self.assertIsNotNone(self.app.lesson_view.cv_image, f"lesson {index + 1}")


if __name__ == "__main__":
    unittest.main()
