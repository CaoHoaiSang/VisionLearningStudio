from __future__ import annotations

import base64
import copy
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from advanced_vision import ALGORITHMS, AdvancedVisionEngine, long_axis_rect, normalize_angle, rotate
from basler_camera import BaslerCamera
from project_model import ProductRepository
from tcp_bridge import BridgeStats, TcpVisionBridge


ROOT = Path(__file__).resolve().parent
COLORS = {
    "page": "#10151B",
    "surface": "#171E26",
    "card": "#1D2630",
    "card_hover": "#24313D",
    "border": "#2C3A47",
    "text": "#EAF1F7",
    "muted": "#8FA1B2",
    "cyan": "#38BDF8",
    "teal": "#2DD4BF",
    "amber": "#F6B84A",
    "danger": "#F0645A",
    "canvas": "#0B1015",
}


def open_image(path: str | Path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


class CVCanvas(tk.Canvas):
    def __init__(self, parent, background=None, **kwargs):
        super().__init__(
            parent,
            bg=background or COLORS["canvas"],
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self.cv_image = None
        self.photo = None
        self.scale = 1.0
        self.offset = (0, 0)
        self._draw_job = None
        self.bind("<Configure>", self._schedule_draw)

    def show(self, image):
        self.cv_image = image
        self._schedule_draw()

    def _schedule_draw(self, _event=None):
        if self._draw_job:
            self.after_cancel(self._draw_job)
        self._draw_job = self.after(20, self.draw)

    def draw(self):
        self._draw_job = None
        self.delete("all")
        if self.cv_image is None:
            self.create_text(
                max(160, self.winfo_width() // 2),
                max(90, self.winfo_height() // 2),
                text="Chưa có ảnh",
                fill=COLORS["muted"],
                font=("Segoe UI", 14),
            )
            return
        image = self.cv_image
        height, width = image.shape[:2]
        canvas_width = max(1, self.winfo_width())
        canvas_height = max(1, self.winfo_height())
        self.scale = min(canvas_width / width, canvas_height / height)
        shown_width = max(1, round(width * self.scale))
        shown_height = max(1, round(height * self.scale))
        resized = cv2.resize(image, (shown_width, shown_height), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(resized[:, :, :3], cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.offset = ((canvas_width - shown_width) // 2, (canvas_height - shown_height) // 2)
        self.create_image(*self.offset, image=self.photo, anchor="nw")

    def point(self, event):
        if self.cv_image is None:
            return None
        u = (event.x - self.offset[0]) / self.scale
        v = (event.y - self.offset[1]) / self.scale
        if 0 <= u < self.cv_image.shape[1] and 0 <= v < self.cv_image.shape[0]:
            return round(u), round(v)
        return None

    def destroy(self):
        if self._draw_job:
            try:
                self.after_cancel(self._draw_job)
            except tk.TclError:
                pass
            self._draw_job = None
        super().destroy()


class LayoutCanvas(tk.Canvas):
    """Native canvas dragging: moving a piece never re-encodes the full scene."""

    def __init__(self, parent, on_select, on_change):
        super().__init__(parent, bg="#DCE5EC", highlightthickness=0, bd=0)
        self.on_select = on_select
        self.on_change = on_change
        self.product = None
        self.loader = None
        self.photos = {}
        self.items = {}
        self.item_keys = {}
        self.drag_key = None
        self.drag_anchor = None
        self.display_scale = 1.0
        self.selected_key = "top"
        self._render_job = None
        self.bind("<Configure>", self._schedule_render)
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

    def set_product(self, product, loader):
        self.product = product
        self.loader = loader
        self._schedule_render()

    def select(self, key):
        self.selected_key = key
        self._update_labels()

    def _schedule_render(self, _event=None):
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(35, self.render)

    def render(self):
        self._render_job = None
        self.delete("all")
        self.photos.clear()
        self.items.clear()
        self.item_keys.clear()
        if not self.product or not self.loader:
            return
        width, height = max(500, self.winfo_width()), max(400, self.winfo_height())
        self.display_scale = min(1.0, width / 1250, height / 760)
        center = (width / 2, height / 2)
        for piece in self.product.pieces:
            sprite = self.loader(piece.template_file)
            mask = self.loader(piece.mask_file)
            if sprite is None or mask is None:
                continue
            rgba = self._rgba(sprite, mask)
            pil = Image.fromarray(rgba).rotate(
                -piece.design_angle,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )
            if self.display_scale != 1:
                pil = pil.resize(
                    (
                        max(1, round(pil.width * self.display_scale)),
                        max(1, round(pil.height * self.display_scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            photo = ImageTk.PhotoImage(pil)
            self.photos[piece.key] = photo
            x = center[0] + piece.design_x * self.display_scale
            y = center[1] + piece.design_y * self.display_scale
            image_item = self.create_image(x, y, image=photo, anchor="center", tags=(piece.key, "piece"))
            label_item = self.create_text(
                x,
                max(24, y - pil.height / 2 - 20),
                text="",
                font=("Segoe UI Semibold", 12),
                tags=(piece.key, "piece-label"),
            )
            self.items[piece.key] = (image_item, label_item, pil.width, pil.height)
            self.item_keys[image_item] = piece.key
            self.item_keys[label_item] = piece.key
        self._update_labels()

    def _update_labels(self):
        if not self.product:
            return
        for piece in self.product.pieces:
            if piece.key not in self.items:
                continue
            color = COLORS["cyan"] if piece.key == self.selected_key else (
                "#23A9E6" if piece.is_top else "#2FBF71"
            )
            label = f"{'MẢNH TRÊN (GẮP)' if piece.is_top else 'MẢNH DƯỚI'}   {piece.design_angle:+.1f}°"
            self.itemconfigure(self.items[piece.key][1], text=label, fill=color)
            self.tag_raise(self.items[piece.key][1])

    def _press(self, event):
        current = self.find_withtag("current")
        if not current:
            return
        key = self.item_keys.get(current[0])
        if not key:
            return
        self.drag_key = key
        self.drag_anchor = (event.x, event.y)
        self.selected_key = key
        self.on_select(key)
        self._update_labels()

    def _drag(self, event):
        if not self.drag_key or not self.drag_anchor:
            return
        dx = event.x - self.drag_anchor[0]
        dy = event.y - self.drag_anchor[1]
        image_item, label_item, _, _ = self.items[self.drag_key]
        self.move(image_item, dx, dy)
        self.move(label_item, dx, dy)
        piece = self.product.piece(self.drag_key)
        piece.design_x += dx / self.display_scale
        piece.design_y += dy / self.display_scale
        self.drag_anchor = (event.x, event.y)
        self.on_change()

    def _release(self, _event):
        self.drag_key = None
        self.drag_anchor = None

    @staticmethod
    def _rgba(sprite, mask):
        bgr = sprite[:, :, :3]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.dstack((rgb, mask))

    def destroy(self):
        if self._render_job:
            try:
                self.after_cancel(self._render_job)
            except tk.TclError:
                pass
            self._render_job = None
        super().destroy()


class VisionLab(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__(fg_color=COLORS["page"])
        self.title("Vision Lab Studio — Gắp & Đặt Vải 2D")
        self.geometry("1680x960")
        self.minsize(1280, 760)

        self.repo = ProductRepository(ROOT / "data" / "products")
        if not self.repo.names():
            self.repo.create("hoc-vision")
        self.settings_path = ROOT / "data" / "settings.json"
        saved = self._read_settings().get("last_product")
        preferred = saved if saved in self.repo.names() else (
            "radxa-live-study" if "radxa-live-study" in self.repo.names() else self.repo.names()[0]
        )
        self.product = self.repo.load(preferred)
        self.image = self.repo.load_image(self.product, self.product.source_image_file)
        self.overlay = self.image
        self.detections = []
        self.selected_piece = tk.StringVar(value="top")
        self.engine = AdvancedVisionEngine(self.product, self._asset)

        self.last_camera_frame = None
        self._bridge_raw = None
        self._bridge_lock = threading.Lock()
        self._bridge_note = ""
        self.events = queue.Queue(maxsize=2)
        self.messages = queue.Queue()
        self.camera = BaslerCamera(self._camera_frame, lambda text: self.messages.put(("camera", text)))
        self.bridge = TcpVisionBridge(
            self._bridge_process,
            self._bridge_frame,
            lambda text, stats: self.messages.put(("bridge", text, BridgeStats(**vars(stats)))),
        )
        self._seg_job = None
        self._tcp_display = {}

        self._build()
        self._load_product_ui()
        self._poll_job = self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self):
        self._header()
        self.tabs = ctk.CTkTabview(
            self,
            fg_color=COLORS["surface"],
            segmented_button_fg_color=COLORS["card"],
            segmented_button_selected_color="#177CAD",
            segmented_button_selected_hover_color="#218FC1",
            segmented_button_unselected_color=COLORS["card"],
            segmented_button_unselected_hover_color=COLORS["card_hover"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"],
        )
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        for name in ("CÀI ĐẶT MẪU", "CAMERA BASLER", "VẬN HÀNH", "CẤU HÌNH TRẠM", "HỌC VISION"):
            self.tabs.add(name)
        self._design_ui(self.tabs.tab("CÀI ĐẶT MẪU"))
        self._camera_ui(self.tabs.tab("CAMERA BASLER"))
        self._operation_ui(self.tabs.tab("VẬN HÀNH"))
        self._station_ui(self.tabs.tab("CẤU HÌNH TRẠM"))
        self._learn_ui(self.tabs.tab("HỌC VISION"))

    def _header(self):
        header = ctk.CTkFrame(self, fg_color="transparent", height=72)
        header.pack(fill="x", padx=18, pady=(12, 8))
        brand = ctk.CTkFrame(header, fg_color="transparent")
        brand.pack(side="left")
        ctk.CTkLabel(
            brand,
            text="VISION LAB",
            font=ctk.CTkFont("Segoe UI", 24, "bold"),
            text_color=COLORS["cyan"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text="Học Vision công nghiệp từ ảnh đến Robot",
            font=ctk.CTkFont("Segoe UI", 12),
            text_color=COLORS["muted"],
        ).pack(anchor="w")

        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right")
        ctk.CTkLabel(actions, text="Sản phẩm", text_color=COLORS["muted"]).pack(side="left", padx=(0, 8))
        self.product_var = tk.StringVar()
        self.product_menu = ctk.CTkOptionMenu(
            actions,
            variable=self.product_var,
            values=self.repo.names(),
            command=lambda _value: self._switch_product(),
            width=230,
            height=36,
            corner_radius=9,
            fg_color=COLORS["card"],
            button_color="#2B6F91",
        )
        self.product_menu.pack(side="left")
        self._button(actions, "Nhập HOXCO", self._import_hoxco, kind="secondary", width=112).pack(
            side="left", padx=8
        )
        self._button(actions, "Thêm", self._new_product, width=76).pack(side="left")
        self._button(actions, "Xóa", self._delete_product, kind="danger", width=72).pack(side="left", padx=(8, 0))

    def _design_ui(self, parent):
        self.design_tabs = ctk.CTkTabview(
            parent,
            fg_color="transparent",
            segmented_button_fg_color=COLORS["card"],
            segmented_button_selected_color="#207DA5",
            corner_radius=12,
        )
        self.design_tabs.pack(fill="both", expand=True, padx=6, pady=6)
        for name in ("B1 · Tách 2 mẫu vải", "B2 · Vị trí tương đối", "B3 · Điểm TCP & ROI"):
            self.design_tabs.add(name)
        self._b1_ui(self.design_tabs.tab("B1 · Tách 2 mẫu vải"))
        self._b2_ui(self.design_tabs.tab("B2 · Vị trí tương đối"))
        self._b3_ui(self.design_tabs.tab("B3 · Điểm TCP & ROI"))

    def _b1_ui(self, parent):
        left = ctk.CTkScrollableFrame(
            parent,
            width=370,
            fg_color=COLORS["surface"],
            scrollbar_button_color="#355166",
            scrollbar_button_hover_color="#47708B",
            corner_radius=12,
        )
        self.b1_scroll = left
        left.pack(side="left", fill="y", padx=(0, 10), pady=6)
        image_card = self._card(parent)
        image_card.pack(side="left", fill="both", expand=True, pady=6)
        self.design_view = CVCanvas(image_card)
        self.design_view.pack(fill="both", expand=True, padx=8, pady=8)
        self.design_view.bind("<Button-1>", self._design_click)

        self._section_title(left, "ẢNH MẪU THIẾT KẾ")
        self._button(left, "Ảnh mẫu hiện tại", self._use_sample, kind="secondary").pack(fill="x", pady=(0, 6))
        self._button(left, "Mở ảnh thiết kế…", self._open_design, kind="secondary").pack(fill="x")

        self._section_title(left, "ĐANG TÁCH MẢNH NÀO?")
        self.piece_selector = ctk.CTkSegmentedButton(
            left,
            values=["Mảnh 1 · DƯỚI", "Mảnh 2 · TRÊN (gắp)"],
            command=self._piece_segment_changed,
            selected_color="#267DA4",
            selected_hover_color="#2B8BB6",
            unselected_color=COLORS["card"],
            unselected_hover_color=COLORS["card_hover"],
            corner_radius=9,
        )
        self.piece_selector.pack(fill="x")
        self.piece_selector.set("Mảnh 2 · TRÊN (gắp)")

        self._section_title(left, "ẢNH NỀN")
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x")
        self._button(row, "Nền từ camera", self._background_from_camera, kind="warning").pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        self._button(row, "Mở ảnh nền…", self._open_background, kind="secondary").pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )
        self.bg_status = tk.StringVar(value="Chưa có ảnh nền")
        ctk.CTkLabel(
            left, textvariable=self.bg_status, text_color=COLORS["teal"], anchor="w", font=("Segoe UI", 12)
        ).pack(fill="x", pady=(6, 0))

        self._section_title(left, "THUẬT TOÁN TÁCH · RIÊNG TỪNG MẢNH")
        self.seg_method = tk.StringVar(value="color")
        method_frame = ctk.CTkFrame(left, fg_color=COLORS["card"], corner_radius=10)
        method_frame.pack(fill="x")
        for label, value in (
            ("Trừ ảnh nền", "background"),
            ("Theo độ sáng", "brightness"),
            ("Theo màu vải", "color"),
        ):
            ctk.CTkRadioButton(
                method_frame,
                text=label,
                variable=self.seg_method,
                value=value,
                command=self._seg_changed,
                fg_color=COLORS["amber"],
                hover_color="#D99A2D",
                border_color="#526273",
            ).pack(anchor="w", padx=12, pady=7)
        self._button(left, "Bấm lên ảnh để lấy màu chuẩn", self._start_color_pick, kind="secondary").pack(
            fill="x", pady=(8, 0)
        )
        self.color_status = tk.StringVar(value="Màu chuẩn: chưa lấy")
        ctk.CTkLabel(left, textvariable=self.color_status, text_color=COLORS["muted"], anchor="w").pack(fill="x")

        self.seg_vars = {
            "blur": tk.IntVar(value=9),
            "threshold": tk.IntVar(value=35),
            "bg_threshold": tk.IntVar(value=20),
            "tol_l": tk.DoubleVar(value=27),
            "tol_ab": tk.DoubleVar(value=10),
            "morph_kernel": tk.IntVar(value=7),
            "min_area_percent": tk.DoubleVar(value=3),
            "pattern_threshold": tk.IntVar(value=30),
        }
        self._section_title(left, "TINH CHỈNH MASK")
        for label, key, minimum, maximum, steps in (
            ("Độ nhạy trừ nền", "bg_threshold", 0, 255, 255),
            ("Làm mượt", "blur", 1, 31, 15),
            ("Ngưỡng sáng thủ công", "threshold", 0, 255, 255),
            ("Dung sai độ sáng L", "tol_l", 1, 80, 79),
            ("Dung sai màu a,b", "tol_ab", 1, 60, 59),
            ("Khử tạp / nhiễu", "morph_kernel", 1, 31, 15),
            ("Diện tích tối thiểu %", "min_area_percent", .1, 20, 199),
            ("Ngưỡng hoa văn", "pattern_threshold", 5, 100, 95),
        ):
            self._slider(left, label, self.seg_vars[key], minimum, maximum, steps, self._seg_changed)
        self.auto_threshold = tk.BooleanVar()
        self.invert = tk.BooleanVar()
        self.merge_fragments = tk.BooleanVar()
        self._switch(left, "Tự động chọn ngưỡng sáng", self.auto_threshold, self._seg_changed)
        self._switch(left, "Nền sáng / vật tối (đảo ngưỡng)", self.invert, self._seg_changed)
        self._switch(left, "Gộp vùng cùng màu khi bị che khuất", self.merge_fragments, self._seg_changed)

        self._section_title(left, "LẤY MẪU")
        self._button(left, "Chạy tách", self._preview_segment).pack(fill="x")
        self._button(left, "Lấy mẫu vào mảnh đang chọn", self._learn_piece, kind="success").pack(
            fill="x", pady=(6, 0)
        )
        self.piece_state = tk.StringVar()
        ctk.CTkLabel(
            left,
            textvariable=self.piece_state,
            text_color=COLORS["muted"],
            justify="left",
            anchor="w",
            wraplength=325,
        ).pack(fill="x", pady=(7, 18))

    def _b2_ui(self, parent):
        side = ctk.CTkFrame(parent, width=300, fg_color=COLORS["surface"], corner_radius=12)
        side.pack(side="right", fill="y", padx=(10, 0), pady=6)
        canvas_card = self._card(parent, color="#DCE5EC")
        canvas_card.pack(side="left", fill="both", expand=True, pady=6)
        self.layout_view = LayoutCanvas(canvas_card, self._layout_selected, self._relation_update)
        self.layout_view.pack(fill="both", expand=True, padx=8, pady=8)

        self._section_title(side, "MẢNH ĐANG CHỌN")
        self.layout_piece_selector = ctk.CTkSegmentedButton(
            side,
            values=["DƯỚI", "TRÊN (gắp)"],
            command=lambda value: self._layout_selected("top" if value.startswith("TRÊN") else "bottom"),
            corner_radius=9,
        )
        self.layout_piece_selector.pack(fill="x", padx=14)
        self.layout_piece_selector.set("TRÊN (gắp)")
        self.layout_angle = tk.DoubleVar()
        self._slider(side, "Góc xoay (độ)", self.layout_angle, -180, 180, 720, self._layout_angle_changed)
        self._section_title(side, "QUAN HỆ THIẾT KẾ")
        self.relation = tk.StringVar()
        ctk.CTkLabel(
            side,
            textvariable=self.relation,
            font=ctk.CTkFont("Consolas", 15),
            text_color=COLORS["teal"],
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=16)
        self._button(side, "Căn lại bố cục", self._fit_layout, kind="secondary").pack(
            fill="x", padx=14, pady=(12, 6)
        )
        self._button(side, "Lưu quan hệ", self._save).pack(fill="x", padx=14)

    def _b3_ui(self, parent):
        toolbar = ctk.CTkFrame(parent, fg_color="transparent")
        toolbar.pack(fill="x", pady=(6, 8))
        ctk.CTkLabel(
            toolbar,
            text="Bấm trực tiếp lên từng ROI để đặt điểm TCP của đầu hút Robot.",
            text_color=COLORS["muted"],
        ).pack(side="left")
        self._button(toolbar, "Làm sạch ROI", self._clean_rois, kind="secondary", width=130).pack(side="right")
        self._button(toolbar, "Lưu mẫu thiết kế", self._save, width=145).pack(side="right", padx=8)
        content = ctk.CTkFrame(parent, fg_color="transparent")
        content.pack(fill="both", expand=True)
        top_card = self._card(content)
        bottom_card = self._card(content)
        top_card.pack(side="left", fill="both", expand=True, padx=(0, 5))
        bottom_card.pack(side="left", fill="both", expand=True, padx=(5, 0))
        ctk.CTkLabel(
            top_card, text="MẢNH 2 · TRÊN (GẮP)", font=("Segoe UI Semibold", 14), text_color=COLORS["cyan"]
        ).pack(anchor="w", padx=14, pady=(10, 3))
        ctk.CTkLabel(
            bottom_card, text="MẢNH 1 · DƯỚI", font=("Segoe UI Semibold", 14), text_color=COLORS["teal"]
        ).pack(anchor="w", padx=14, pady=(10, 3))
        self.tcp_top = CVCanvas(top_card, background="#E7EDF2")
        self.tcp_bottom = CVCanvas(bottom_card, background="#E7EDF2")
        self.tcp_top.pack(fill="both", expand=True, padx=10, pady=8)
        self.tcp_bottom.pack(fill="both", expand=True, padx=10, pady=8)
        self.tcp_top.bind("<Button-1>", lambda event: self._tcp_click("top", self.tcp_top, event))
        self.tcp_bottom.bind("<Button-1>", lambda event: self._tcp_click("bottom", self.tcp_bottom, event))
        self.tcp_text = {"top": tk.StringVar(), "bottom": tk.StringVar()}
        top_footer = ctk.CTkFrame(top_card, fg_color="transparent")
        top_footer.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(top_footer, textvariable=self.tcp_text["top"], text_color=COLORS["muted"]).pack(
            side="left"
        )
        self._button(
            top_footer, "TCP về tâm vật", lambda: self._tcp_to_object_center("top"),
            kind="secondary", width=132,
        ).pack(side="right")
        bottom_footer = ctk.CTkFrame(bottom_card, fg_color="transparent")
        bottom_footer.pack(fill="x", padx=14, pady=(0, 10))
        ctk.CTkLabel(
            bottom_footer, textvariable=self.tcp_text["bottom"], text_color=COLORS["muted"]
        ).pack(side="left")
        self._button(
            bottom_footer, "TCP về tâm vật", lambda: self._tcp_to_object_center("bottom"),
            kind="secondary", width=132,
        ).pack(side="right")

    def _camera_ui(self, parent):
        side = ctk.CTkScrollableFrame(parent, width=340, fg_color=COLORS["surface"], corner_radius=12)
        side.pack(side="left", fill="y", padx=(0, 10), pady=8)
        camera_card = self._card(parent)
        camera_card.pack(side="left", fill="both", expand=True, pady=8)
        self.camera_view = CVCanvas(camera_card)
        self.camera_view.pack(fill="both", expand=True, padx=8, pady=8)

        self._section_title(side, "KẾT NỐI CAMERA BASLER")
        self.device = tk.StringVar()
        self.device_menu = ctk.CTkOptionMenu(side, variable=self.device, values=["Chưa quét"], height=38)
        self.device_menu.pack(fill="x")
        self._button(side, "Quét Basler", self._scan, kind="secondary").pack(fill="x", pady=(6, 0))
        self._button(side, "Kết nối", self._connect).pack(fill="x", pady=(6, 0))
        self.camera_status = tk.StringVar(value="Chưa kết nối")
        ctk.CTkLabel(
            side,
            textvariable=self.camera_status,
            text_color=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=305,
        ).pack(fill="x", pady=6)

        self._section_title(side, "THÔNG SỐ CHỤP")
        self.exposure = tk.DoubleVar(value=10000)
        self.gain = tk.DoubleVar(value=0)
        self.camera_width = tk.IntVar(value=0)
        self.camera_height = tk.IntVar(value=0)
        self.camera_fps = tk.DoubleVar(value=20)
        self.trigger_mode = tk.StringVar(value="Off")
        for label, variable in (
            ("Exposure (µs)", self.exposure),
            ("Gain (dB)", self.gain),
            ("ROI Width (0 = giữ nguyên)", self.camera_width),
            ("ROI Height (0 = giữ nguyên)", self.camera_height),
            ("Giới hạn FPS", self.camera_fps),
        ):
            self._entry(side, label, variable)
        ctk.CTkLabel(side, text="Trigger", text_color=COLORS["muted"], anchor="w").pack(fill="x", pady=(5, 2))
        ctk.CTkOptionMenu(side, variable=self.trigger_mode, values=["Off", "Software", "Line1"]).pack(fill="x")
        self._button(side, "Áp dụng thông số", self._apply_camera).pack(fill="x", pady=(8, 0))
        self._button(side, "LIVE / DỪNG · LatestImageOnly", self._toggle_camera, kind="success").pack(
            fill="x", pady=(6, 0)
        )
        self._button(side, "Phát Software Trigger", self._software_trigger, kind="secondary").pack(
            fill="x", pady=(6, 0)
        )
        self.cam_overlay = tk.BooleanVar()
        self._switch(side, "Hiển thị overlay phát hiện", self.cam_overlay)
        self._section_title(side, "DÙNG ẢNH ĐÃ CHỤP")
        self._button(side, "Chụp → Cài đặt mẫu", self._camera_design, kind="secondary").pack(fill="x")
        self._button(side, "Chụp → Vận hành", self._camera_operation, kind="warning").pack(
            fill="x", pady=(6, 14)
        )

    def _operation_ui(self, parent):
        side = ctk.CTkScrollableFrame(parent, width=330, fg_color=COLORS["surface"], corner_radius=12)
        side.pack(side="left", fill="y", padx=(0, 10), pady=8)
        right = ctk.CTkFrame(parent, width=350, fg_color=COLORS["surface"], corner_radius=12)
        right.pack(side="right", fill="y", padx=(10, 0), pady=8)
        center = self._card(parent)
        center.pack(side="left", fill="both", expand=True, pady=8)
        self.run_view = CVCanvas(center)
        self.run_view.pack(fill="both", expand=True, padx=8, pady=8)

        self._section_title(side, "NGUỒN ẢNH")
        self.source_mode = tk.StringVar(value="Ảnh tĩnh")
        self.source_selector = ctk.CTkSegmentedButton(
            side,
            variable=self.source_mode,
            values=["Ảnh tĩnh", "Robot · DeltaX"],
            command=self._source_mode_changed,
            corner_radius=9,
        )
        self.source_selector.pack(fill="x")
        self.static_source = ctk.CTkFrame(side, fg_color="transparent")
        self.static_source.pack(fill="x", pady=(8, 0))
        self._button(self.static_source, "Ảnh mẫu sản phẩm", self._use_sample, kind="secondary").pack(fill="x")
        self._button(self.static_source, "Mở ảnh hiện trường…", self._open_scene, kind="secondary").pack(
            fill="x", pady=(6, 0)
        )
        self._button(self.static_source, "Lấy frame mới nhất từ Basler", self._use_camera_scene, kind="secondary").pack(
            fill="x", pady=(6, 0)
        )
        self.robot_source = ctk.CTkFrame(side, fg_color="transparent")
        self.bridge_button = self._button(self.robot_source, "MỞ CẦU NỐI ROBOT", self._toggle_bridge, kind="success")
        self.bridge_button.pack(fill="x")
        self.bridge_status = tk.StringVar(value="Chưa mở")
        self.bridge_stats = tk.StringVar(value="Khách: -\nKhung: 0\nNhịp: 0 fps")
        ctk.CTkLabel(
            self.robot_source, textvariable=self.bridge_status, text_color=COLORS["muted"], anchor="w"
        ).pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(
            self.robot_source, textvariable=self.bridge_stats, text_color=COLORS["teal"], anchor="w",
            justify="left", font=("Consolas", 12)
        ).pack(fill="x")

        self._section_title(side, "PHÁT HIỆN")
        self.ghost = tk.BooleanVar()
        self._switch(side, "Xem trước vị trí ĐẶT (ghost)", self.ghost, self._detect)
        self._button(side, "CHẠY — TÍNH GẮP & ĐẶT", self._detect).pack(fill="x", pady=(5, 0))
        self._button(side, "Lưu cảnh làm mẫu train", self._capture, kind="secondary").pack(
            fill="x", pady=(6, 0)
        )
        self._button(side, "Xuất lệnh Robot (.txt)", self._export, kind="danger").pack(
            fill="x", pady=(6, 14)
        )

        self._section_title(right, "KẾT QUẢ")
        self.result = tk.StringVar(value="Chưa chạy")
        ctk.CTkLabel(
            right,
            textvariable=self.result,
            text_color=COLORS["teal"],
            anchor="w",
            justify="left",
            wraplength=315,
        ).pack(fill="x", padx=14)
        self._section_title(right, "LỆNH GỬI ROBOT")
        self.command = ctk.CTkTextbox(
            right, width=325, fg_color=COLORS["canvas"], border_width=1, border_color=COLORS["border"],
            font=("Consolas", 12), corner_radius=10
        )
        self.command.pack(fill="both", expand=True, padx=12)
        self._section_title(right, "NHẬT KÝ")
        self.log = ctk.CTkTextbox(
            right, height=150, fg_color=COLORS["canvas"], border_width=1, border_color=COLORS["border"],
            font=("Consolas", 11), corner_radius=10
        )
        self.log.pack(fill="x", padx=12, pady=(0, 12))
        self._source_mode_changed("Ảnh tĩnh")

    def _station_ui(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=10)
        station_card = self._card(scroll)
        station_card.pack(fill="x", pady=(0, 10))
        self._section_title(station_card, "HẰNG SỐ TRẠM")
        self.stvars = {
            "port": tk.IntVar(value=6001),
            "allow_lan": tk.BooleanVar(value=True),
            "angle_sign": tk.DoubleVar(value=-1),
            "angle_offset": tk.DoubleVar(value=0),
            "mm_per_pixel": tk.DoubleVar(value=.25),
            "origin_u": tk.DoubleVar(),
            "origin_v": tk.DoubleVar(),
            "max_result_age_ms": tk.IntVar(value=1500),
            "scale_tolerance_percent": tk.DoubleVar(value=25),
        }
        grid = ctk.CTkFrame(station_card, fg_color="transparent")
        grid.pack(fill="x", padx=16, pady=(0, 10))
        entries = (
            ("Cổng cầu nối", "port"),
            ("Dấu góc trục W", "angle_sign"),
            ("Bù góc (°)", "angle_offset"),
            ("mm / pixel", "mm_per_pixel"),
            ("Gốc U", "origin_u"),
            ("Gốc V", "origin_v"),
            ("Tuổi kết quả tối đa (ms)", "max_result_age_ms"),
        )
        for index, (label, key) in enumerate(entries):
            cell = ctk.CTkFrame(grid, fg_color="transparent")
            cell.grid(row=index // 3, column=index % 3, padx=7, pady=5, sticky="ew")
            grid.grid_columnconfigure(index % 3, weight=1)
            self._entry(cell, label, self.stvars[key])
        self._switch(station_card, "Nhận ảnh từ mạng LAN", self.stvars["allow_lan"])

        algorithm_card = self._card(scroll)
        algorithm_card.pack(fill="x", pady=(0, 10))
        self._section_title(algorithm_card, "THUẬT TOÁN ĐỊNH VỊ")
        self.algorithm_label = tk.StringVar(value=ALGORITHMS["auto"])
        self.algorithm_menu = ctk.CTkOptionMenu(
            algorithm_card,
            variable=self.algorithm_label,
            values=list(ALGORITHMS.values()),
            width=410,
            height=38,
        )
        self.algorithm_menu.pack(anchor="w", padx=16)
        self._entry(algorithm_card, "Dung sai tỉ lệ ± %", self.stvars["scale_tolerance_percent"])

        calibration_card = self._card(scroll)
        calibration_card.pack(fill="x", pady=(0, 10))
        self._section_title(calibration_card, "HIỆU CHUẨN MẶT PHẲNG · PIXEL → ROBOT")
        ctk.CTkLabel(
            calibration_card,
            text="Mỗi dòng: u, v, X, Y · cần ít nhất 4 điểm không thẳng hàng",
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=16)
        row = ctk.CTkFrame(calibration_card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=10)
        self.calibration_points = ctk.CTkTextbox(
            row, height=110, fg_color=COLORS["canvas"], border_width=1, border_color=COLORS["border"],
            font=("Consolas", 12), corner_radius=10
        )
        self.calibration_points.pack(side="left", fill="x", expand=True)
        actions = ctk.CTkFrame(row, fg_color="transparent", width=260)
        actions.pack(side="left", fill="y", padx=(10, 0))
        self._button(actions, "Tính Homography", self._calibrate).pack(fill="x")
        self._button(actions, "Xóa Homography", self._clear_calibration, kind="danger").pack(fill="x", pady=6)
        self.calibration_status = tk.StringVar(value="Chưa hiệu chuẩn · đang dùng mm/pixel")
        ctk.CTkLabel(
            actions, textvariable=self.calibration_status, text_color=COLORS["muted"], wraplength=250,
            justify="left"
        ).pack(anchor="w")
        self._button(scroll, "LƯU CẤU HÌNH TRẠM", self._save_station, width=210).pack(anchor="e", pady=(0, 10))

    def _learn_ui(self, parent):
        left = self._card(parent)
        left.pack(side="left", fill="y", padx=(0, 10), pady=8)
        right = self._card(parent)
        right.pack(side="left", fill="both", expand=True, pady=8)
        self._section_title(left, "LỘ TRÌNH 15 BÀI")
        self.lessons = tk.Listbox(
            left,
            width=36,
            bg=COLORS["canvas"],
            fg=COLORS["text"],
            selectbackground="#227CA4",
            selectforeground="white",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 11),
        )
        self.lesson_names = [
            "Pixel, BGR, Gray và LAB", "Histogram & ánh sáng", "Threshold và mask", "Morphology",
            "Contour và minAreaRect", "Tâm, góc và TCP", "Trừ ảnh nền", "Template matching ảnh xám",
            "ORB và Homography", "Che khuất và biến dạng", "Pixel → mm", "Quan hệ PICK/PLACE",
            "Basler Exposure/Gain", "TCP bridge và frame age", "An toàn Robot",
        ]
        for index, name in enumerate(self.lesson_names, 1):
            self.lessons.insert("end", f"{index:02d} · {name}")
        self.lessons.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.lessons.bind("<<ListboxSelect>>", self._lesson)
        lesson_header = ctk.CTkFrame(right, fg_color="transparent")
        lesson_header.pack(fill="x", padx=12, pady=(10, 0))
        self.lesson_title = tk.StringVar()
        ctk.CTkLabel(
            lesson_header, textvariable=self.lesson_title,
            font=ctk.CTkFont("Segoe UI", 16, "bold"), text_color=COLORS["cyan"],
        ).pack(side="left")
        self.lesson_piece_key = tk.StringVar(value="TRÊN (gắp)")
        self.lesson_piece_selector = ctk.CTkSegmentedButton(
            lesson_header,
            variable=self.lesson_piece_key,
            values=["DƯỚI", "TRÊN (gắp)"],
            command=self._lesson_piece_changed,
            width=250,
            corner_radius=9,
        )
        self.lesson_piece_selector.pack(side="right")
        self.lesson_text = ctk.CTkTextbox(
            right, height=92, fg_color=COLORS["canvas"], border_width=1, border_color=COLORS["border"],
            font=("Segoe UI", 13), corner_radius=10, wrap="word"
        )
        self.lesson_text.pack(fill="x", padx=12, pady=(8, 6))
        self.lesson_controls = ctk.CTkFrame(
            right, height=150, fg_color=COLORS["surface"], corner_radius=10,
            border_width=1, border_color=COLORS["border"],
        )
        self.lesson_controls.pack(fill="x", padx=12, pady=(0, 6))
        self.lesson_controls.pack_propagate(False)
        self.lesson_view = CVCanvas(right)
        self.lesson_view.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.lesson_vars = {}
        self.lesson_index = 0
        self._lesson_job = None
        self.lessons.selection_set(0)
        self._lesson(None)

    def _card(self, parent, color=None):
        return ctk.CTkFrame(
            parent,
            fg_color=color or COLORS["card"],
            corner_radius=14,
            border_width=1,
            border_color=COLORS["border"] if color is None else color,
        )

    def _section_title(self, parent, text):
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
            text_color=COLORS["cyan"],
            anchor="w",
        ).pack(fill="x", padx=2, pady=(16, 7))

    def _button(self, parent, text, command, kind="primary", width=0):
        palette = {
            "primary": ("#188AB8", "#229DCE"),
            "secondary": ("#2A3946", "#354958"),
            "success": ("#169B83", "#20AD94"),
            "warning": ("#B97A1A", "#D08C25"),
            "danger": ("#C9504A", "#DF5E57"),
        }
        fg, hover = palette[kind]
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=36,
            corner_radius=9,
            fg_color=fg,
            hover_color=hover,
            font=ctk.CTkFont("Segoe UI", 12, "bold"),
        )

    def _entry(self, parent, label, variable):
        ctk.CTkLabel(parent, text=label, text_color=COLORS["muted"], anchor="w").pack(
            fill="x", padx=2, pady=(5, 2)
        )
        ctk.CTkEntry(
            parent,
            textvariable=variable,
            height=34,
            corner_radius=8,
            fg_color=COLORS["canvas"],
            border_color=COLORS["border"],
        ).pack(fill="x", padx=2)

    def _slider(self, parent, label, variable, minimum, maximum, steps, callback):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=2, pady=5)
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=label, text_color=COLORS["text"], anchor="w").pack(side="left")
        value_label = ctk.CTkLabel(top, text="", text_color=COLORS["amber"], width=52, anchor="e")
        value_label.pack(side="right")

        def changed(value):
            display = f"{value:.1f}" if isinstance(variable, tk.DoubleVar) else f"{round(value)}"
            value_label.configure(text=display)
            callback()

        slider = ctk.CTkSlider(
            row,
            from_=minimum,
            to=maximum,
            number_of_steps=steps,
            variable=variable,
            command=changed,
            progress_color=COLORS["amber"],
            button_color=COLORS["amber"],
            button_hover_color="#FFD27A",
            fg_color="#3A4650",
            height=16,
        )
        slider.pack(fill="x")
        initial = variable.get()
        value_label.configure(text=f"{initial:.1f}" if isinstance(variable, tk.DoubleVar) else str(initial))
        return slider

    def _switch(self, parent, text, variable, command=None):
        switch = ctk.CTkSwitch(
            parent,
            text=text,
            variable=variable,
            command=command,
            progress_color=COLORS["teal"],
            button_color="#EAF7F5",
            button_hover_color="white",
        )
        switch.pack(anchor="w", padx=2, pady=5)
        return switch

    def _read_settings(self):
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _asset(self, filename):
        return self.repo.load_image(self.product, filename)

    def _piece(self):
        return self.product.piece(self.selected_piece.get())

    def _load_product_ui(self):
        self.product_menu.configure(values=self.repo.names())
        self.product_var.set(self.product.name)
        self.image = self._asset(self.product.source_image_file)
        self.overlay = self.image
        self.design_view.show(self.image)
        self.run_view.show(self.overlay)
        self.layout_view.set_product(self.product, self._asset)
        self.layout_angle.set(self.product.piece("top").design_angle)
        self.layout_piece_selector.set("TRÊN (gắp)")
        self._piece_to_ui()
        self._relation_update()
        self._tcp_render()
        background = self._asset(self.product.background_file)
        self.bg_status.set(
            f"Đã có ảnh nền {background.shape[1]}×{background.shape[0]}" if background is not None else "Chưa có ảnh nền"
        )
        for key, variable in self.stvars.items():
            variable.set(getattr(self.product.station, key))
        self.algorithm_label.set(ALGORITHMS.get(self.product.station.algorithm, ALGORITHMS["auto"]))
        self.calibration_status.set(
            "Homography 3×3 đã sẵn sàng" if self.product.station.homography else
            "Chưa hiệu chuẩn · đang dùng mm/pixel"
        )
        if hasattr(self, "lesson_controls"):
            self._lesson_piece_changed()
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(
            json.dumps({"last_product": self.product.name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _new_product(self):
        name = simpledialog.askstring("Thêm sản phẩm", "Tên sản phẩm mới\n(mẫu hiện tại được dùng làm bản sao):")
        if not name:
            return
        try:
            self.product = self.repo.create(name, self.product)
            self.engine.update_product(self.product, self._asset)
            self._load_product_ui()
        except Exception as exc:
            messagebox.showerror("Vision Lab", str(exc))

    def _import_hoxco(self):
        folder = filedialog.askdirectory(title="Chọn thư mục sản phẩm HOXCO có template.json")
        if not folder:
            return
        name = simpledialog.askstring(
            "Nhập sản phẩm HOXCO",
            "Tên sản phẩm mới:",
            initialvalue=f"{Path(folder).name}-import",
        )
        if not name:
            return
        try:
            self.product = self.repo.import_hoxco(folder, name)
            self.engine.update_product(self.product, self._asset)
            self._load_product_ui()
            self._log("Đã nhập bản sao sản phẩm HOXCO; dữ liệu gốc không bị thay đổi.")
        except Exception as exc:
            messagebox.showerror("Nhập HOXCO", str(exc))

    def _delete_product(self):
        if len(self.repo.names()) <= 1:
            return messagebox.showinfo("Vision Lab", "Phải giữ ít nhất một sản phẩm.")
        if not messagebox.askyesno("Xóa sản phẩm", f"Xóa '{self.product.name}'? Không thể khôi phục."):
            return
        self.repo.delete(self.product.name)
        self.product = self.repo.load(self.repo.names()[0])
        self.engine.update_product(self.product, self._asset)
        self._load_product_ui()

    def _switch_product(self):
        name = self.product_var.get()
        if name and name != self.product.name:
            self.product = self.repo.load(name)
            self.engine.update_product(self.product, self._asset)
            self._load_product_ui()

    def _piece_segment_changed(self, value):
        self.selected_piece.set("top" if value.startswith("Mảnh 2") else "bottom")
        self._piece_to_ui()

    def _piece_to_ui(self):
        piece = self._piece()
        self.seg_method.set(piece.params.get("method", "color"))
        for key, variable in self.seg_vars.items():
            variable.set(piece.params.get(key, variable.get()))
        self.auto_threshold.set(piece.params.get("auto_threshold", False))
        self.invert.set(piece.params.get("invert", False))
        self.merge_fragments.set(piece.params.get("merge_fragments", False))
        self.color_status.set(
            "Màu chuẩn LAB: " + ", ".join(str(round(value)) for value in piece.reference_lab)
            if piece.reference_lab else "Màu chuẩn: chưa lấy"
        )
        self.piece_state.set(
            "\n".join(
                f"{item.name}: {'đã có mẫu' if item.template_file else 'chưa lấy mẫu'}"
                for item in self.product.pieces
            )
        )

    def _seg_changed(self):
        piece = self._piece()
        piece.params["method"] = self.seg_method.get()
        piece.params["auto_threshold"] = self.auto_threshold.get()
        piece.params["invert"] = self.invert.get()
        piece.params["merge_fragments"] = self.merge_fragments.get()
        for key, variable in self.seg_vars.items():
            piece.params[key] = variable.get()
        if self.image is not None:
            if self._seg_job:
                self.after_cancel(self._seg_job)
            self._seg_job = self.after(140, self._preview_segment)

    def _use_sample(self):
        image = self._asset(self.product.source_image_file)
        if image is not None:
            self.image = image
            self.design_view.show(image)
            self.run_view.show(image)

    def _open_design(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image = open_image(path)
            self.product.source_image_file = self.repo.save_image(self.product, "design_source.jpg", self.image)
            self.repo.save(self.product)
            self.design_view.show(self.image)

    def _start_color_pick(self):
        self._color_pick = True
        self.color_status.set("Bấm vào vùng giữa của mảnh vải…")

    def _design_click(self, event):
        point = self.design_view.point(event)
        if point and self.image is not None and getattr(self, "_color_pick", False):
            color = self.engine.sample_color(self.image, *point, self._piece())
            self._color_pick = False
            self.color_status.set("Màu chuẩn LAB: " + ", ".join(str(round(value)) for value in color))
            self._preview_segment()

    def _background_from_camera(self):
        if self.last_camera_frame is None:
            return messagebox.showinfo("Ảnh nền", "Chưa có frame Basler. Hãy bật LIVE hoặc chụp camera trước.")
        self._save_background(self.last_camera_frame)

    def _open_background(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh nền", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self._save_background(open_image(path))

    def _save_background(self, image):
        if image is None:
            return
        self.product.background_file = self.repo.save_image(self.product, "background.png", image)
        self.repo.save(self.product)
        self.bg_status.set(f"Đã có ảnh nền {image.shape[1]}×{image.shape[0]}")
        self._log("Đã cập nhật ảnh nền.")

    def _preview_segment(self):
        self._seg_job = None
        if self.image is None:
            return
        piece = self._piece()
        mask, contours = self.engine.segment_piece(self.image, piece)
        shown = self.image.copy()
        cv2.drawContours(shown, contours, -1, tuple(piece.draw_color_bgr), 3, cv2.LINE_AA)
        self.design_view.show(shown)
        self.lesson_view.show(mask)

    def _learn_piece(self):
        if self.image is None:
            return
        try:
            piece = self._piece()
            sprite, mask, _ = self.engine.learn_piece(self.image, piece)
            piece.template_file = self.repo.save_image(self.product, f"{piece.key}_template.png", sprite)
            piece.mask_file = self.repo.save_image(self.product, f"{piece.key}_mask.png", mask)
            self.repo.save(self.product)
            self._piece_to_ui()
            self.layout_view.set_product(self.product, self._asset)
            self._tcp_render()
        except Exception as exc:
            messagebox.showerror("Lấy mẫu", str(exc))

    def _layout_selected(self, key):
        self.selected_piece.set(key)
        self.layout_piece_selector.set("TRÊN (gắp)" if key == "top" else "DƯỚI")
        self.layout_angle.set(self.product.piece(key).design_angle)
        self.layout_view.select(key)
        self._relation_update()

    def _layout_angle_changed(self):
        self.product.piece(self.selected_piece.get()).design_angle = self.layout_angle.get()
        self.layout_view.render()
        self._relation_update()

    def _relation_update(self):
        bottom, top = self.product.piece("bottom"), self.product.piece("top")
        bottom_axis = self._mask_axis_angle(bottom) + bottom.design_angle
        top_axis = self._mask_axis_angle(top) + top.design_angle
        self.relation.set(
            f"Δx = {top.design_x-bottom.design_x:+.0f} px\n"
            f"Δy = {top.design_y-bottom.design_y:+.0f} px\n"
            f"Δθ thanh trượt = {top.design_angle-bottom.design_angle:+.1f}°\n"
            f"Δθ thực trên mẫu = {normalize_angle(top_axis-bottom_axis):+.1f}°"
        )

    def _fit_layout(self):
        bottom, top = self.product.piece("bottom"), self.product.piece("top")
        bottom.design_x, bottom.design_y = 0, 120
        top.design_x, top.design_y = 0, -120
        self.layout_view.render()
        self._relation_update()

    def _masked_roi(self, key):
        piece = self.product.piece(key)
        sprite = self._asset(piece.template_file)
        mask = self._asset(piece.mask_file)
        if sprite is None or mask is None:
            return None, (0, 0), None
        mask = self.engine.clean_mask(mask)
        points = cv2.findNonZero(mask)
        if points is None:
            return sprite, (0, 0), mask
        x, y, width, height = cv2.boundingRect(points)
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(sprite.shape[1], x + width + pad), min(sprite.shape[0], y + height + pad)
        crop = sprite[y0:y1, x0:x1, :3].copy()
        crop_mask = mask[y0:y1, x0:x1]
        neutral = np.full_like(crop, (232, 237, 241))
        alpha = (crop_mask.astype(np.float32) / 255)[:, :, None]
        shown = (neutral * (1 - alpha) + crop * alpha).astype(np.uint8)
        return shown, (x0, y0), crop_mask

    def _tcp_render(self):
        for key, view in (("top", self.tcp_top), ("bottom", self.tcp_bottom)):
            piece = self.product.piece(key)
            shown, origin, _ = self._masked_roi(key)
            if shown is None:
                view.show(None)
                continue
            original = self._asset(piece.template_file)
            old_center = (original.shape[1] / 2, original.shape[0] / 2)
            point = (
                round(old_center[0] + piece.tcp_offset_local[0] - origin[0]),
                round(old_center[1] + piece.tcp_offset_local[1] - origin[1]),
            )
            cv2.drawMarker(shown, point, (30, 45, 235), cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
            cv2.putText(shown, "TCP", (point[0] + 9, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 45, 235), 2)
            view.show(shown)
            self._tcp_display[key] = (origin, original.shape[:2])
            self.tcp_text[key].set(
                f"TCP: ({piece.tcp_offset_local[0]:+.0f}, {piece.tcp_offset_local[1]:+.0f}) px so với tâm"
            )

    def _tcp_click(self, key, view, event):
        point = view.point(event)
        if not point or key not in self._tcp_display:
            return
        origin, shape = self._tcp_display[key]
        original_u, original_v = point[0] + origin[0], point[1] + origin[1]
        self.product.piece(key).tcp_offset_local = [
            original_u - shape[1] / 2,
            original_v - shape[0] / 2,
        ]
        self._tcp_render()

    def _tcp_to_object_center(self, key):
        piece = self.product.piece(key)
        mask = self._asset(piece.mask_file)
        if mask is None:
            return messagebox.showinfo("TCP", "Mảnh này chưa có mask. Hãy lấy mẫu ở B1 trước.")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = self.engine.clean_mask(mask)
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] <= 0:
            return messagebox.showinfo("TCP", "Mask không có vùng vật hợp lệ.")
        center_u = moments["m10"] / moments["m00"]
        center_v = moments["m01"] / moments["m00"]
        piece.tcp_offset_local = [
            float(center_u - mask.shape[1] / 2),
            float(center_v - mask.shape[0] / 2),
        ]
        self._tcp_render()
        self._log(f"Đã đặt TCP {piece.name} tại tâm hình học của mask.")

    def _clean_rois(self):
        changed = 0
        for piece in self.product.pieces:
            sprite = self._asset(piece.template_file)
            mask = self._asset(piece.mask_file)
            if sprite is None or mask is None:
                continue
            clean = self.engine.clean_mask(mask)
            points = cv2.findNonZero(clean)
            if points is None:
                continue
            x, y, width, height = cv2.boundingRect(points)
            pad = 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(sprite.shape[1], x + width + pad), min(sprite.shape[0], y + height + pad)
            old_center = np.array([sprite.shape[1] / 2, sprite.shape[0] / 2])
            new_center_in_old = np.array([x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2])
            piece.tcp_offset_local = (
                np.asarray(piece.tcp_offset_local) + old_center - new_center_in_old
            ).astype(float).tolist()
            piece.template_file = self.repo.save_image(
                self.product, f"{piece.key}_template.png", sprite[y0:y1, x0:x1, :3]
            )
            piece.mask_file = self.repo.save_image(
                self.product, f"{piece.key}_mask.png", clean[y0:y1, x0:x1]
            )
            changed += 1
        self.repo.save(self.product)
        self.layout_view.set_product(self.product, self._asset)
        self._tcp_render()
        self._log(f"Đã làm sạch {changed} ROI và giữ nguyên vị trí TCP thực.")

    def _save(self):
        self.repo.save(self.product)
        self._log("Đã lưu mẫu thiết kế.")

    def _scan(self):
        try:
            if not self.camera.available:
                self.camera_status.set("Chưa cài pypylon/Basler pylon Runtime")
                return
            devices = self.camera.enumerate()
            values = [f"{item['model']} | {item['serial']} | {item['ip']}" for item in devices]
            self.device_menu.configure(values=values or ["Không tìm thấy camera"])
            self.device.set(values[0] if values else "Không tìm thấy camera")
            self.camera_status.set(f"Tìm thấy {len(devices)} camera Basler")
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _connect(self):
        try:
            serial = self.device.get().split("|")[1].strip() if "|" in self.device.get() else None
            self.camera.connect(serial)
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _apply_camera(self):
        try:
            values = self.camera.apply(
                self.exposure.get(), self.gain.get(), self.camera_width.get(), self.camera_height.get(),
                self.camera_fps.get(), self.trigger_mode.get()
            )
            self.camera_status.set(
                f"{values.get('Width', 0)}×{values.get('Height', 0)} · "
                f"Exposure={values.get('ExposureTime', 0):.0f}µs · Gain={values.get('Gain', 0):.1f}dB · "
                f"Trigger={values.get('TriggerMode', 'Off')}"
            )
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _toggle_camera(self):
        try:
            self.camera.stop() if self.camera.running else self.camera.start()
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _software_trigger(self):
        try:
            self.camera.software_trigger()
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _camera_frame(self, frame, fps):
        self._latest(("camera", frame, fps))

    def _camera_design(self):
        if self.last_camera_frame is None:
            return messagebox.showinfo("Camera", "Chưa có frame Basler.")
        self.image = self.last_camera_frame.copy()
        self.product.source_image_file = self.repo.save_image(self.product, "design_source.jpg", self.image)
        self.repo.save(self.product)
        self.design_view.show(self.image)
        self.tabs.set("CÀI ĐẶT MẪU")

    def _camera_operation(self):
        if self.last_camera_frame is None:
            return messagebox.showinfo("Camera", "Chưa có frame Basler.")
        self.image = self.last_camera_frame.copy()
        self.run_view.show(self.image)
        self.tabs.set("VẬN HÀNH")

    def _source_mode_changed(self, value):
        if value == "Ảnh tĩnh":
            self.robot_source.pack_forget()
            self.static_source.pack(fill="x", pady=(8, 0))
        else:
            self.static_source.pack_forget()
            self.robot_source.pack(fill="x", pady=(8, 0))

    def _open_scene(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image = open_image(path)
            self.run_view.show(self.image)

    def _use_camera_scene(self):
        if self.last_camera_frame is None:
            return messagebox.showinfo("Camera", "Chưa có frame Basler.")
        self.image = self.last_camera_frame.copy()
        self.run_view.show(self.image)

    def _detect(self):
        if self.image is None:
            return
        self._station_to_model()
        self.detections, self.overlay, debug = self.engine.process(self.image, self.product.station.algorithm)
        if self.ghost.get():
            self.overlay = self._ghost_overlay(self.overlay, self.detections)
        self.run_view.show(self.overlay)
        self._show_result(debug.elapsed_ms, debug.notes)

    def _show_result(self, elapsed, notes):
        objects = {item.type_id: item for item in self.detections}
        pick, place = objects.get(1), objects.get(0)
        if not pick or not place:
            self.result.set(f"Không định vị đủ hai mảnh\nXử lý: {elapsed:.1f} ms")
            text = "// Không khớp đủ 2 mảnh."
        else:
            station = self.product.station
            pick_xy = self._pixel_to_robot(pick.tcp_u, pick.tcp_v)
            place_xy = self._pixel_to_robot(place.tcp_u, place.tcp_v)
            pick_w = station.angle_sign * pick.angle + station.angle_offset
            place_w = station.angle_sign * place.angle + station.angle_offset
            self.result.set(
                f"Khớp mẫu thành công\n"
                f"TRÊN: ({pick.tcp_u:.0f},{pick.tcp_v:.0f}) {pick.angle:+.1f}° · {pick.score:.2f}\n"
                f"DƯỚI: ({place.tcp_u:.0f},{place.tcp_v:.0f}) {place.angle:+.1f}° · {place.score:.2f}\n"
                f"Xử lý: {elapsed:.1f} ms"
            )
            text = (
                f"// ===== VISION LAB -> ROBOT =====\n\n"
                f"PICK:\n  u,v = {pick.tcp_u:.0f}, {pick.tcp_v:.0f} px\n"
                f"  theta = {pick.angle:+.2f}°\n  X,Y = {pick_xy[0]:.1f}, {pick_xy[1]:.1f} mm\n"
                f"  W = {pick_w:+.2f}°\n\n"
                f"PLACE:\n  u,v = {place.tcp_u:.0f}, {place.tcp_v:.0f} px\n"
                f"  theta = {place.angle:+.2f}°\n  X,Y = {place_xy[0]:.1f}, {place_xy[1]:.1f} mm\n"
                f"  W = {place_w:+.2f}°\n\nROTATE = {place_w-pick_w:+.2f}°"
            )
        self.command.delete("1.0", "end")
        self.command.insert("1.0", text)
        for note in notes:
            self._log(note)

    def _ghost_overlay(self, image, detections):
        bottom_detection = next((item for item in detections if item.type_id == 0), None)
        top_piece, bottom_piece = self.product.piece("top"), self.product.piece("bottom")
        sprite, mask = self._asset(top_piece.template_file), self._asset(top_piece.mask_file)
        if bottom_detection is None or sprite is None or mask is None:
            return image
        target_u, target_v, sprite_rotation = self._ghost_transform(bottom_detection)
        # Detection/B2 use positive angles clockwise in image coordinates. OpenCV
        # warpAffine uses positive angles counter-clockwise, hence the minus sign.
        sprite, mask = self._rotate_sprite(sprite[:, :, :3], mask, -sprite_rotation)
        x = round(target_u - sprite.shape[1] / 2)
        y = round(target_v - sprite.shape[0] / 2)
        composed = image.copy()
        self._paste(composed, sprite, mask, x, y)
        ghost = cv2.addWeighted(image, .58, composed, .42, 0)
        cv2.putText(
            ghost, "VI TRI DAT THEO THIET KE", (max(4, x), max(24, y)),
            cv2.FONT_HERSHEY_SIMPLEX, .58, (30, 50, 255), 2, cv2.LINE_AA
        )
        return ghost

    def _mask_axis_angle(self, piece):
        mask = self._asset(piece.mask_file)
        if mask is None:
            return 0.0
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        return long_axis_rect(max(contours, key=cv2.contourArea))[2]

    def _ghost_transform(self, bottom_detection):
        """Map the exact B2 composition onto the detected bottom piece."""
        top_piece = self.product.piece("top")
        bottom_piece = self.product.piece("bottom")
        bottom_design_axis = self._mask_axis_angle(bottom_piece) + bottom_piece.design_angle
        scene_rotation = normalize_angle(bottom_detection.angle - bottom_design_axis)
        delta = (
            top_piece.design_x - bottom_piece.design_x,
            top_piece.design_y - bottom_piece.design_y,
        )
        dx, dy = rotate(delta, scene_rotation)
        sprite_rotation = normalize_angle(top_piece.design_angle + scene_rotation)
        return (
            bottom_detection.center_u + dx,
            bottom_detection.center_v + dy,
            sprite_rotation,
        )

    def _bridge_process(self, image):
        with self._bridge_lock:
            self._bridge_raw = image.copy()
        started = time.perf_counter()
        detections, overlay, _ = self.engine.process(image, self.product.station.algorithm)
        elapsed = (time.perf_counter() - started) * 1000
        maximum = max(1, int(self.product.station.max_result_age_ms))
        self._bridge_note = ""
        if elapsed > maximum:
            detections = []
            self._bridge_note = f"Từ chối kết quả: xử lý {elapsed:.0f} ms > giới hạn {maximum} ms"
        return detections, overlay

    def _bridge_frame(self, overlay, detections, stats):
        self._latest(("bridge", overlay, detections, BridgeStats(**vars(stats))))

    def _toggle_bridge(self):
        if self.bridge.running:
            self.bridge.stop()
            self.bridge_button.configure(text="MỞ CẦU NỐI ROBOT", fg_color="#169B83", hover_color="#20AD94")
        else:
            self._station_to_model()
            host = "0.0.0.0" if self.product.station.allow_lan else "127.0.0.1"
            self.bridge.start(host, self.product.station.port)
            self.bridge_button.configure(text="ĐÓNG CẦU NỐI ROBOT", fg_color="#C9504A", hover_color="#DF5E57")

    def _capture(self):
        if self.image is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = self.repo.capture_folder(self.product)
        self.repo.save_image(self.product, f"captures/scene_{stamp}.jpg", self.image)
        (folder / f"scene_{stamp}.json").write_text(
            json.dumps(
                {
                    "time": stamp,
                    "algorithm": self.product.station.algorithm,
                    "objects": [
                        {"type": item.type_id, "tcp": [item.tcp_u, item.tcp_v], "angle": item.angle, "score": item.score}
                        for item in self.detections
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._log(f"Đã lưu cảnh scene_{stamp}.jpg")

    def _export(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            Path(path).write_text(self.command.get("1.0", "end"), encoding="utf-8")

    def _station_to_model(self):
        for key, variable in self.stvars.items():
            setattr(self.product.station, key, variable.get())
        selected_label = self.algorithm_label.get()
        self.product.station.algorithm = next(
            (key for key, label in ALGORITHMS.items() if label == selected_label), "auto"
        )

    def _save_station(self):
        self._station_to_model()
        self.repo.save(self.product)
        self._log("Đã lưu cấu hình trạm.")

    def _calibrate(self):
        image_points, robot_points = [], []
        try:
            for number, raw in enumerate(self.calibration_points.get("1.0", "end").splitlines(), 1):
                if not raw.strip():
                    continue
                values = [
                    float(part.strip())
                    for part in raw.replace("=>", ",").replace(";", ",").split(",")
                    if part.strip()
                ]
                if len(values) != 4:
                    raise ValueError(f"Dòng {number} phải có đúng 4 số: u, v, X, Y")
                image_points.append(values[:2])
                robot_points.append(values[2:])
            if len(image_points) < 4:
                raise ValueError("Cần ít nhất 4 cặp điểm hiệu chuẩn")
            matrix, _ = cv2.findHomography(
                np.asarray(image_points, np.float32), np.asarray(robot_points, np.float32), cv2.RANSAC, 1.5
            )
            if matrix is None:
                raise ValueError("Không tính được homography; kiểm tra các điểm có thẳng hàng không")
            projected = cv2.perspectiveTransform(
                np.asarray(image_points, np.float32).reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((projected - np.asarray(robot_points)) ** 2, axis=1))))
            self.product.station.homography = matrix.reshape(-1).astype(float).tolist()
            self.calibration_status.set(f"Đã hiệu chuẩn {len(image_points)} điểm · RMS {rms:.3f} mm")
        except Exception as exc:
            messagebox.showerror("Hiệu chuẩn", str(exc))

    def _clear_calibration(self):
        self.product.station.homography = None
        self.calibration_status.set("Chưa hiệu chuẩn · đang dùng mm/pixel")

    def _pixel_to_robot(self, u, v):
        matrix = self.product.station.homography
        if matrix and len(matrix) == 9:
            point = cv2.perspectiveTransform(
                np.asarray([[[float(u), float(v)]]], np.float32),
                np.asarray(matrix, np.float64).reshape(3, 3),
            )[0, 0]
            return float(point[0]), float(point[1])
        station = self.product.station
        return (
            (float(u) - station.origin_u) * station.mm_per_pixel,
            (float(v) - station.origin_v) * station.mm_per_pixel,
        )

    def _lesson(self, _event):
        selected = self.lessons.curselection()
        if not selected:
            return
        self.lesson_index = selected[0]
        lessons = [
            "LAB tách độ sáng L khỏi màu a,b, phù hợp hơn BGR khi ánh sáng dao động.",
            "Exposure quá cao làm mất texture; quá thấp tăng nhiễu và mất điểm đặc trưng.",
            "Mask tốt có vật trắng, nền đen, ít lỗ và ít vùng giả.",
            "Opening bỏ nhiễu; Closing lấp lỗ. Kernel quá lớn sẽ làm biến dạng biên.",
            "Contour mô tả biên; minAreaRect trả tâm, kích thước và góc cạnh dài.",
            "TCP là điểm hút. Offset TCP phải xoay theo pose vật.",
            "Trừ nền mạnh khi camera và ánh sáng cố định; phải lấy nền lại khi trạm đổi.",
            "Ảnh xám/edge matching dễ hiểu nhưng phải quét góc và scale.",
            "ORB + Homography chịu xoay, scale và che khuất khi vật có hoa văn.",
            "Vải biến dạng cần kết hợp màu, contour và feature thay vì một tiêu chí.",
            "mm/pixel là mô hình cơ bản; Homography cần bốn điểm chuẩn trở lên.",
            "Δx, Δy, Δθ lưu quan hệ thiết kế giữa mảnh gắp và mảnh đặt.",
            "Basler cần tắt auto trước khi đặt Exposure/Gain; LatestImageOnly chống frame cũ.",
            "Bridge chỉ nên giữ frame mới nhất và từ chối kết quả quá tuổi.",
            "Thử Robot với Z an toàn, tốc độ thấp và đầu hút tắt trước.",
        ]
        self.lesson_title.set(f"{self.lesson_index + 1:02d} · {self.lesson_names[self.lesson_index]}")
        self.lesson_text.delete("1.0", "end")
        self.lesson_text.insert("1.0", lessons[self.lesson_index])
        self._build_lesson_controls(self.lesson_index)
        self._schedule_lesson_render()

    def _lesson_piece_changed(self, _value=None):
        self._build_lesson_controls(self.lesson_index)
        self._schedule_lesson_render()

    def _lesson_piece(self):
        return self.product.piece("top" if self.lesson_piece_key.get().startswith("TRÊN") else "bottom")

    def _lesson_cell(self):
        body = self.lesson_control_body
        index = len(body.winfo_children())
        column = index % 4
        row = index // 4
        body.grid_columnconfigure(column, weight=1)
        cell = ctk.CTkFrame(body, fg_color="transparent")
        cell.grid(row=row, column=column, sticky="ew", padx=6, pady=3)
        return cell

    def _lesson_slider(self, key, label, initial, minimum, maximum, steps, integer=False):
        variable = tk.IntVar(value=round(initial)) if integer else tk.DoubleVar(value=float(initial))
        self.lesson_vars[key] = variable
        self._slider(
            self._lesson_cell(), label, variable, minimum, maximum, steps,
            self._schedule_lesson_render,
        )
        return variable

    def _lesson_option(self, key, label, values, initial):
        cell = self._lesson_cell()
        ctk.CTkLabel(cell, text=label, text_color=COLORS["muted"], anchor="w").pack(fill="x", pady=(5, 2))
        variable = tk.StringVar(value=initial)
        self.lesson_vars[key] = variable
        ctk.CTkOptionMenu(
            cell, variable=variable, values=values,
            command=lambda _value: self._schedule_lesson_render(),
            height=34, corner_radius=8,
        ).pack(fill="x")
        return variable

    def _lesson_switch(self, key, label, initial=False):
        variable = tk.BooleanVar(value=bool(initial))
        self.lesson_vars[key] = variable
        self._switch(self._lesson_cell(), label, variable, self._schedule_lesson_render)
        return variable

    def _build_lesson_controls(self, index):
        for child in self.lesson_controls.winfo_children():
            child.destroy()
        self.lesson_vars = {}
        self.lesson_control_body = ctk.CTkFrame(self.lesson_controls, fg_color="transparent")
        self.lesson_control_body.pack(fill="both", expand=True, padx=8, pady=5)
        piece = self._lesson_piece()
        params = piece.params
        source = self.image if self.image is not None else self._asset(self.product.source_image_file)
        height, width = source.shape[:2] if source is not None else (720, 1280)
        if index == 0:
            self._lesson_option("channel", "Kênh hiển thị", ["BGR", "Gray", "LAB · L", "LAB · a", "LAB · b"], "BGR")
        elif index == 1:
            self._lesson_slider("brightness", "Độ sáng cộng", 0, -120, 120, 240, True)
            self._lesson_slider("contrast", "Tương phản", 1, .2, 2.5, 230)
        elif index == 2:
            self._lesson_slider("threshold", "Ngưỡng 0–255", params.get("threshold", 130), 0, 255, 255, True)
            self._lesson_slider("blur", "Làm mượt", params.get("blur", 9), 1, 31, 30, True)
            self._lesson_switch("invert", "Vật tối / đảo ngưỡng", params.get("invert", False))
            self._lesson_switch("auto", "Otsu tự chọn ngưỡng", params.get("auto_threshold", False))
        elif index == 3:
            self._lesson_slider("threshold", "Ngưỡng tạo mask", params.get("threshold", 130), 0, 255, 255, True)
            self._lesson_slider("kernel", "Kích thước kernel", params.get("morph_kernel", 7), 1, 31, 30, True)
            self._lesson_slider("iterations", "Số lần xử lý", 1, 1, 5, 4, True)
            self._lesson_option("operation", "Phép hình thái", ["Opening", "Closing"], "Opening")
        elif index == 4:
            self._lesson_slider("tol_l", "Dung sai sáng L", params.get("tol_l", 27), 1, 80, 79)
            self._lesson_slider("tol_ab", "Dung sai màu a,b", params.get("tol_ab", 10), 1, 60, 59)
            self._lesson_slider("min_area", "Diện tích tối thiểu %", params.get("min_area_percent", 3), .1, 20, 199)
        elif index == 5:
            self._lesson_slider("tcp_x", "TCP lệch X (px)", piece.tcp_offset_local[0], -250, 250, 500)
            self._lesson_slider("tcp_y", "TCP lệch Y (px)", piece.tcp_offset_local[1], -250, 250, 500)
        elif index == 6:
            self._lesson_slider("bg_threshold", "Độ nhạy trừ nền", params.get("bg_threshold", 20), 0, 255, 255, True)
            self._lesson_slider("blur", "Làm mượt sai khác", params.get("blur", 9), 1, 31, 30, True)
            self._lesson_switch("auto", "Otsu tự chọn ngưỡng", params.get("auto_threshold", False))
        elif index == 7:
            self._lesson_slider("angle", "Góc mẫu (độ)", 0, -90, 90, 180)
            self._lesson_slider("scale", "Tỉ lệ mẫu", 1, .5, 1.5, 100)
            self._lesson_slider("edge", "Ngưỡng cạnh Canny", 60, 10, 180, 170, True)
        elif index == 8:
            self._lesson_slider("features", "Số đặc trưng tối đa", 900, 100, 2000, 190, True)
            self._lesson_slider("fast", "Độ nhạy FAST", 8, 1, 60, 59, True)
        elif index == 9:
            self._lesson_slider("occlusion", "Che khuất (%)", 30, 0, 85, 85)
            self._lesson_slider("angle", "Góc vật (độ)", 0, -90, 90, 180)
        elif index == 10:
            self._lesson_slider("u", "Pixel u", width / 2, 0, max(1, width - 1), max(1, width - 1), True)
            self._lesson_slider("v", "Pixel v", height / 2, 0, max(1, height - 1), max(1, height - 1), True)
            self._lesson_slider("mmpp", "mm / pixel", self.product.station.mm_per_pixel, .01, 2, 199)
        elif index == 11:
            bottom, top = self.product.piece("bottom"), self.product.piece("top")
            self._lesson_slider("dx", "Δx (px)", top.design_x - bottom.design_x, -400, 400, 800)
            self._lesson_slider("dy", "Δy (px)", top.design_y - bottom.design_y, -300, 300, 600)
            self._lesson_slider("dtheta", "Δθ (độ)", top.design_angle - bottom.design_angle, -90, 90, 360)
        elif index == 12:
            self._lesson_slider("exposure", "Exposure mô phỏng", 1, .2, 2.5, 230)
            self._lesson_slider("gain", "Gain / nhiễu (dB)", 0, 0, 24, 120)
        elif index == 13:
            self._lesson_slider("age", "Tuổi frame (ms)", 200, 0, 4000, 400, True)
            self._lesson_slider("max_age", "Giới hạn chấp nhận (ms)", self.product.station.max_result_age_ms, 100, 4000, 390, True)
        else:
            self._lesson_slider("speed", "Tốc độ chạy thử (%)", 10, 1, 100, 99)
            self._lesson_slider("clearance", "Khoảng hở Z mô phỏng (mm)", 50, 0, 150, 150)

    def _schedule_lesson_render(self):
        if getattr(self, "_lesson_job", None):
            try:
                self.after_cancel(self._lesson_job)
            except tk.TclError:
                pass
        self._lesson_job = self.after(55, self._render_current_lesson)

    def _lesson_source(self):
        source = self.image if self.image is not None else self._asset(self.product.source_image_file)
        if source is None:
            return np.full((720, 1280, 3), (18, 25, 32), np.uint8)
        return source[:, :, :3].copy()

    @staticmethod
    def _lesson_caption(image, text, color=(245, 245, 245)):
        shown = image.copy()
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 42), (15, 22, 29), -1)
        cv2.putText(shown, text, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
        return shown

    @staticmethod
    def _lesson_pair(left, right, left_label="TRƯỚC", right_label="SAU"):
        target_h = max(left.shape[0], right.shape[0])
        def fit(image):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            scale = target_h / image.shape[0]
            return cv2.resize(image, (max(1, round(image.shape[1] * scale)), target_h))
        a, b = fit(left), fit(right)
        cv2.putText(a, left_label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(b, right_label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
        return np.hstack((a, b))

    def _render_current_lesson(self):
        self._lesson_job = None
        source = self._lesson_source()
        piece = self._lesson_piece()
        values = {key: variable.get() for key, variable in self.lesson_vars.items()}
        index = self.lesson_index
        shown = source
        if index == 0:
            mode = values["channel"]
            if mode == "Gray":
                shown = cv2.cvtColor(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            elif mode.startswith("LAB"):
                lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
                channel = {"LAB · L": 0, "LAB · a": 1, "LAB · b": 2}[mode]
                shown = cv2.cvtColor(lab[:, :, channel], cv2.COLOR_GRAY2BGR)
            shown = self._lesson_caption(shown, f"Kenh dang xem: {mode}")
        elif index == 1:
            shown = cv2.convertScaleAbs(source, alpha=float(values["contrast"]), beta=int(values["brightness"]))
            gray = cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
            histogram = histogram / max(1, histogram.max()) * 125
            x0, y0, chart_w = 20, shown.shape[0] - 25, min(520, shown.shape[1] - 40)
            cv2.rectangle(shown, (10, y0 - 145), (x0 + chart_w + 10, y0 + 8), (15, 22, 29), -1)
            points = np.array([
                [x0 + round(i * chart_w / 255), y0 - round(value)]
                for i, value in enumerate(histogram)
            ], np.int32)
            cv2.polylines(shown, [points], False, (80, 220, 245), 2, cv2.LINE_AA)
            shown = self._lesson_caption(
                shown, f"Brightness {values['brightness']:+.0f} · Contrast {values['contrast']:.2f}"
            )
        elif index == 2:
            blur = max(1, int(values["blur"]))
            blur += 1 - blur % 2
            gray = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
            flag = cv2.THRESH_BINARY_INV if values["invert"] else cv2.THRESH_BINARY
            if values["auto"]:
                flag |= cv2.THRESH_OTSU
            threshold = 0 if values["auto"] else int(values["threshold"])
            used, mask = cv2.threshold(gray, threshold, 255, flag)
            shown = self._lesson_pair(source, mask, "ANH GOC", f"MASK - NGUONG {used:.0f}")
        elif index == 3:
            gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, int(values["threshold"]), 255, cv2.THRESH_BINARY)
            kernel_size = max(1, int(values["kernel"]))
            kernel_size += 1 - kernel_size % 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            operation = cv2.MORPH_OPEN if values["operation"] == "Opening" else cv2.MORPH_CLOSE
            result = cv2.morphologyEx(mask, operation, kernel, iterations=max(1, int(values["iterations"])))
            shown = self._lesson_pair(mask, result, "MASK BAN DAU", values["operation"].upper())
        elif index == 4:
            probe = copy.deepcopy(piece)
            probe.params["method"] = "color"
            probe.params["tol_l"] = values["tol_l"]
            probe.params["tol_ab"] = values["tol_ab"]
            probe.params["min_area_percent"] = values["min_area"]
            _mask, contours = self.engine.segment_piece(source, probe)
            shown = source.copy()
            for contour in contours:
                cv2.drawContours(shown, [contour], -1, tuple(piece.draw_color_bgr), 3, cv2.LINE_AA)
                box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
                cv2.polylines(shown, [box], True, (40, 210, 245), 2, cv2.LINE_AA)
            shown = self._lesson_caption(shown, f"Tim thay {len(contours)} contour hop le")
        elif index == 5:
            sprite = self._asset(piece.template_file)
            mask = self._asset(piece.mask_file)
            if sprite is not None and mask is not None:
                shown = sprite[:, :, :3].copy()
                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                moments = cv2.moments(mask, binaryImage=True)
                center = (
                    round(moments["m10"] / moments["m00"]),
                    round(moments["m01"] / moments["m00"]),
                ) if moments["m00"] else (shown.shape[1] // 2, shown.shape[0] // 2)
                tcp = (
                    round(shown.shape[1] / 2 + values["tcp_x"]),
                    round(shown.shape[0] / 2 + values["tcp_y"]),
                )
                cv2.drawMarker(shown, center, (60, 220, 80), cv2.MARKER_CROSS, 32, 2)
                cv2.drawMarker(shown, tcp, (20, 40, 245), cv2.MARKER_TILTED_CROSS, 34, 3)
                shown = self._lesson_caption(shown, f"Tam mask xanh - TCP do ({values['tcp_x']:+.0f}, {values['tcp_y']:+.0f}) px")
        elif index == 6:
            background = self._asset(self.product.background_file)
            if background is None:
                shown = self._lesson_caption(source, "Chua co anh nen - lay nen o B1 truoc", (80, 180, 255))
            else:
                background = cv2.resize(background[:, :, :3], (source.shape[1], source.shape[0]))
                diff = np.max(cv2.absdiff(source, background), axis=2)
                blur = max(1, int(values["blur"]))
                blur += 1 - blur % 2
                diff = cv2.GaussianBlur(diff, (blur, blur), 0)
                flag = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if values["auto"] else 0)
                threshold = 0 if values["auto"] else int(values["bg_threshold"])
                used, mask = cv2.threshold(diff, threshold, 255, flag)
                shown = self._lesson_pair(
                    cv2.applyColorMap(diff, cv2.COLORMAP_TURBO), mask,
                    "DO KHAC NEN", f"MASK - NGUONG {used:.0f}",
                )
        elif index == 7:
            sprite = self._asset(piece.template_file)
            mask = self._asset(piece.mask_file)
            if sprite is not None:
                scale = float(values["scale"])
                resized = cv2.resize(sprite[:, :, :3], None, fx=scale, fy=scale)
                use_mask = mask if mask is not None else np.full(sprite.shape[:2], 255, np.uint8)
                if use_mask.ndim == 3:
                    use_mask = cv2.cvtColor(use_mask, cv2.COLOR_BGR2GRAY)
                use_mask = cv2.resize(use_mask, (resized.shape[1], resized.shape[0]))
                rotated, _ = self._rotate_sprite(resized, use_mask, -float(values["angle"]))
                edge = int(values["edge"])
                edges = cv2.Canny(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), edge, min(255, edge * 2))
                shown = self._lesson_pair(rotated, edges, "MAU XOAY / SCALE", "CANH DUNG SO KHOP")
        elif index == 8:
            target = self._asset(piece.template_file)
            target = target[:, :, :3] if target is not None else source
            gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(
                nfeatures=max(50, int(values["features"])),
                fastThreshold=max(1, int(values["fast"])),
            )
            keypoints = orb.detect(gray, None)
            shown = cv2.drawKeypoints(
                target, keypoints, None, color=(40, 210, 245),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            shown = self._lesson_caption(shown, f"ORB phat hien {len(keypoints)} diem dac trung")
        elif index == 9:
            sprite = self._asset(piece.template_file)
            mask = self._asset(piece.mask_file)
            if sprite is not None:
                use_mask = mask if mask is not None else np.full(sprite.shape[:2], 255, np.uint8)
                if use_mask.ndim == 3:
                    use_mask = cv2.cvtColor(use_mask, cv2.COLOR_BGR2GRAY)
                rotated, _ = self._rotate_sprite(sprite[:, :, :3], use_mask, -float(values["angle"]))
                hidden = rotated.copy()
                cut = round(hidden.shape[1] * float(values["occlusion"]) / 100)
                if cut:
                    cv2.rectangle(hidden, (hidden.shape[1] - cut, 0), (hidden.shape[1], hidden.shape[0]), (28, 35, 43), -1)
                shown = self._lesson_pair(rotated, hidden, "VAT DAY DU", f"CHE {values['occlusion']:.0f}%")
        elif index == 10:
            u, v = int(values["u"]), int(values["v"])
            mmpp = float(values["mmpp"])
            station = self.product.station
            x = (u - station.origin_u) * mmpp
            y = (v - station.origin_v) * mmpp
            shown = source.copy()
            cv2.drawMarker(shown, (u, v), (20, 40, 245), cv2.MARKER_CROSS, 34, 3)
            shown = self._lesson_caption(shown, f"({u}, {v}) px -> ({x:.2f}, {y:.2f}) mm - {mmpp:.3f} mm/px")
        elif index == 11:
            shown = np.full((650, 1000, 3), (25, 33, 42), np.uint8)
            bottom = self.product.piece("bottom")
            top = self.product.piece("top")
            bottom_sprite, bottom_mask = self._asset(bottom.template_file), self._asset(bottom.mask_file)
            top_sprite, top_mask = self._asset(top.template_file), self._asset(top.mask_file)
            center = (500, 330)
            if bottom_sprite is not None and bottom_mask is not None:
                self._paste(shown, bottom_sprite[:, :, :3], bottom_mask, center[0] - bottom_sprite.shape[1] // 2, center[1] - bottom_sprite.shape[0] // 2)
            if top_sprite is not None and top_mask is not None:
                rotated, rotated_mask = self._rotate_sprite(top_sprite[:, :, :3], top_mask, -float(values["dtheta"]))
                x = round(center[0] + values["dx"] - rotated.shape[1] / 2)
                y = round(center[1] + values["dy"] - rotated.shape[0] / 2)
                self._paste(shown, rotated, rotated_mask, x, y)
            shown = self._lesson_caption(shown, f"DX {values['dx']:+.0f}px - DY {values['dy']:+.0f}px - DTHETA {values['dtheta']:+.1f} deg")
        elif index == 12:
            exposure = float(values["exposure"])
            gain = float(values["gain"])
            rng = np.random.default_rng(42)
            simulated = source.astype(np.float32) * exposure
            simulated += rng.normal(0, gain * 1.6, simulated.shape)
            simulated = np.clip(simulated, 0, 255).astype(np.uint8)
            shown = self._lesson_pair(source, simulated, "ẢNH GỐC", f"EXPOSURE {exposure:.2f} · GAIN {gain:.1f}")
        elif index == 13:
            age, maximum = int(values["age"]), int(values["max_age"])
            accepted = age <= maximum
            color = (60, 220, 90) if accepted else (40, 60, 245)
            shown = (source.astype(np.float32) * .45).astype(np.uint8)
            width = max(1, shown.shape[1] - 80)
            ratio = min(1, age / max(1, maximum))
            cv2.rectangle(shown, (40, shown.shape[0] // 2), (40 + width, shown.shape[0] // 2 + 28), (55, 65, 75), -1)
            cv2.rectangle(shown, (40, shown.shape[0] // 2), (40 + round(width * ratio), shown.shape[0] // 2 + 28), color, -1)
            status = "CHAP NHAN KET QUA" if accepted else "TU CHOI FRAME QUA TUOI"
            shown = self._lesson_caption(shown, f"{status} · age {age} ms / max {maximum} ms", color)
        else:
            shown = (source.astype(np.float32) * .5).astype(np.uint8)
            shown = self._lesson_caption(
                shown,
                f"MO PHONG - khong gui Robot - toc do {values['speed']:.0f}% - khoang ho Z {values['clearance']:.0f} mm",
                (80, 210, 245),
            )
        self.lesson_view.show(shown)

    def _latest(self, item):
        try:
            self.events.put_nowait(item)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(item)

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "camera":
                    _, frame, fps = event
                    self.last_camera_frame = frame.copy()
                    shown = frame
                    if self.cam_overlay.get():
                        self.detections, shown, _ = self.engine.process(frame, self.product.station.algorithm)
                    self.camera_view.show(shown)
                    self.camera_status.set(f"LIVE · {fps:.1f} fps · frame mới nhất")
                else:
                    _, self.overlay, self.detections, stats = event
                    with self._bridge_lock:
                        if self._bridge_raw is not None:
                            self.image = self._bridge_raw.copy()
                    if self.ghost.get():
                        self.overlay = self._ghost_overlay(self.overlay, self.detections)
                    self.run_view.show(self.overlay)
                    self.bridge_stats.set(
                        f"Khách: {stats.client}\nKhung: {stats.frames}\nNhịp: {stats.fps:.1f} fps\n"
                        f"Xử lý: {stats.processing_ms:.1f} ms"
                    )
                    self._show_result(stats.processing_ms, [self._bridge_note] if self._bridge_note else [])
        except queue.Empty:
            pass
        try:
            while True:
                event = self.messages.get_nowait()
                if event[0] == "camera":
                    self.camera_status.set(event[1])
                else:
                    _, text, _stats = event
                    self.bridge_status.set(text)
                    self._log(text)
        except queue.Empty:
            pass
        self._poll_job = self.after(50, self._poll)

    def _log(self, text):
        if hasattr(self, "log"):
            self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.log.see("end")

    @staticmethod
    def _rotate_sprite(sprite, mask, angle):
        height, width = sprite.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width, new_height = int(height * sine + width * cosine), int(height * cosine + width * sine)
        matrix[0, 2] += new_width / 2 - width / 2
        matrix[1, 2] += new_height / 2 - height / 2
        return (
            cv2.warpAffine(sprite, matrix, (new_width, new_height)),
            cv2.warpAffine(mask, matrix, (new_width, new_height)),
        )

    @staticmethod
    def _paste(canvas, sprite, mask, x, y):
        height, width = sprite.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas.shape[1], x + width), min(canvas.shape[0], y + height)
        if x0 >= x1 or y0 >= y1:
            return
        sx, sy = x0 - x, y0 - y
        crop = sprite[sy:sy + y1 - y0, sx:sx + x1 - x0]
        alpha = (mask[sy:sy + y1 - y0, sx:sx + x1 - x0].astype(float) / 255)[:, :, None]
        canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1 - alpha) + crop * alpha).astype(np.uint8)

    def _close(self):
        for job in (
            getattr(self, "_poll_job", None),
            getattr(self, "_seg_job", None),
            getattr(self, "_lesson_job", None),
        ):
            if job:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        self.camera.close()
        self.bridge.stop()
        self.repo.save(self.product)
        self.destroy()


if __name__ == "__main__":
    VisionLab().mainloop()
