from __future__ import annotations

import base64
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np

from advanced_vision import ALGORITHMS, AdvancedVisionEngine
from basler_camera import BaslerCamera
from project_model import ProductRepository
from tcp_bridge import BridgeStats, TcpVisionBridge


ROOT = Path(__file__).resolve().parent


def open_image(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


class ImageView(tk.Canvas):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", "#0a0d11")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, **kwargs)
        self.image = self.photo = None
        self.scale, self.offset = 1.0, (0, 0)
        self.bind("<Configure>", lambda _e: self.draw())

    def show(self, image):
        self.image = image
        self.draw()

    def draw(self):
        self.delete("all")
        if self.image is None:
            self.create_text(max(150, self.winfo_width() // 2), max(80, self.winfo_height() // 2),
                             text="Chưa có ảnh", fill="#768594", font=("Segoe UI", 15))
            return
        h, w = self.image.shape[:2]
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        self.scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        image = cv2.resize(self.image, (nw, nh), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        ok, data = cv2.imencode(".png", image)
        if not ok:
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(data).decode())
        self.offset = ((cw - nw) // 2, (ch - nh) // 2)
        self.create_image(*self.offset, image=self.photo, anchor="nw")

    def point(self, event):
        if self.image is None:
            return None
        u = (event.x - self.offset[0]) / self.scale
        v = (event.y - self.offset[1]) / self.scale
        return (round(u), round(v)) if 0 <= u < self.image.shape[1] and 0 <= v < self.image.shape[0] else None


class VisionLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision Lab Studio — Gắp & Đặt Vải 2D")
        self.geometry("1600x920")
        self.minsize(1200, 740)
        self.configure(bg="#171b20")
        self._theme()
        self.repo = ProductRepository(ROOT / "data" / "products")
        if not self.repo.names():
            self.repo.create("hoc-vision")
        self.settings_path = ROOT / "data" / "settings.json"
        saved_name = None
        try:
            saved_name = json.loads(self.settings_path.read_text(encoding="utf-8")).get("last_product")
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        preferred = saved_name if saved_name in self.repo.names() else \
            ("radxa-live-study" if "radxa-live-study" in self.repo.names() else self.repo.names()[0])
        self.product = self.repo.load(preferred)
        self.image = self.overlay = self.repo.load_image(self.product, self.product.source_image_file)
        self._bridge_raw = None
        self._bridge_lock = threading.Lock()
        self._bridge_note = ""
        self.detections, self.click_mode = [], None
        self.selected_piece = tk.StringVar(value="top")
        self.engine = AdvancedVisionEngine(self.product, self._asset)
        self.events = queue.Queue(maxsize=2)
        self.messages = queue.Queue()
        self.camera = BaslerCamera(self._camera_frame, lambda text: self.messages.put(("camera", text)))
        self.bridge = TcpVisionBridge(self._bridge_process, self._bridge_frame,
                                      lambda text, stats: self.messages.put(("bridge", text, BridgeStats(**vars(stats)))))
        self._build()
        self._load_ui()
        self.after(50, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background="#20252b", foreground="#e8edf2", fieldbackground="#11161b")
        s.configure("TFrame", background="#20252b")
        s.configure("TLabel", background="#20252b", foreground="#e8edf2")
        s.configure("TLabelframe", background="#20252b", foreground="#24bdf5")
        s.configure("TLabelframe.Label", background="#20252b", foreground="#24bdf5")
        s.configure("TButton", background="#08bd93", foreground="#061510", padding=(9, 6))
        s.configure("Danger.TButton", background="#e64a42", foreground="white")
        s.configure("Blue.TButton", background="#397cae", foreground="white")
        s.configure("TNotebook", background="#171b20", borderwidth=0)
        s.configure("TNotebook.Tab", background="#315f88", foreground="white", padding=(14, 8))
        s.map("TNotebook.Tab", background=[("selected", "#171b20")])

    def _build(self):
        header = ttk.Frame(self, padding=(14, 8))
        header.pack(fill="x")
        ttk.Label(header, text="VISION LAB", font=("Segoe UI Semibold", 19), foreground="#13a9db").pack(side="left")
        ttk.Label(header, text=" · Học Vision công nghiệp từ ảnh đến Robot").pack(side="left")
        ttk.Button(header, text="Xóa", style="Danger.TButton", command=self._delete_product).pack(side="right")
        ttk.Button(header, text="Thêm", command=self._new_product).pack(side="right", padx=5)
        ttk.Button(header, text="Nhập HOXCO", style="Blue.TButton", command=self._import_hoxco).pack(side="right")
        self.product_var = tk.StringVar()
        self.products = ttk.Combobox(header, textvariable=self.product_var, state="readonly", width=27)
        self.products.pack(side="right", padx=6)
        self.products.bind("<<ComboboxSelected>>", lambda _e: self._switch_product())
        ttk.Label(header, text="Sản phẩm:").pack(side="right")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.design, self.camtab, self.run, self.station, self.learn = [ttk.Frame(self.tabs) for _ in range(5)]
        for tab, title in zip((self.design, self.camtab, self.run, self.station, self.learn),
                              ("CÀI ĐẶT MẪU THIẾT KẾ", "CAMERA BASLER", "VẬN HÀNH",
                               "CẤU HÌNH TRẠM", "HỌC VISION")):
            self.tabs.add(tab, text=title)
        self._design_ui()
        self._camera_ui()
        self._run_ui()
        self._station_ui()
        self._learn_ui()

    def _design_ui(self):
        sub = ttk.Notebook(self.design)
        sub.pack(fill="both", expand=True)
        self.b1, self.b2, self.b3 = [ttk.Frame(sub) for _ in range(3)]
        for tab, title in zip((self.b1, self.b2, self.b3),
                              ("B1 · Tách 2 mẫu vải", "B2 · Vị trí tương đối (kéo–thả)", "B3 · Điểm TCP Robot & Lưu")):
            sub.add(tab, text=title)
        self._b1_ui()
        self._b2_ui()
        self._b3_ui()

    def _b1_ui(self):
        side = ttk.Frame(self.b1, width=340, padding=10)
        side.pack(side="left", fill="y")
        self.design_view = ImageView(self.b1)
        self.design_view.pack(fill="both", expand=True, padx=8, pady=8)
        self.design_view.bind("<Button-1>", self._design_click)
        box = ttk.Labelframe(side, text="Ảnh mẫu thiết kế", padding=8)
        box.pack(fill="x")
        ttk.Button(box, text="Mở ảnh…", style="Blue.TButton", command=self._open_design).pack(fill="x")
        ttk.Button(box, text="Lấy ảnh hiện tại làm nền", command=self._set_background).pack(fill="x", pady=4)
        box = ttk.Labelframe(side, text="Đang tách mảnh nào?", padding=8)
        box.pack(fill="x", pady=8)
        for label, key in (("Mảnh 1 · DƯỚI", "bottom"), ("Mảnh 2 · TRÊN (gắp)", "top")):
            ttk.Radiobutton(box, text=label, variable=self.selected_piece, value=key,
                            command=self._piece_to_ui).pack(anchor="w")
        box = ttk.Labelframe(side, text="Thuật toán tách · riêng từng mảnh", padding=8)
        box.pack(fill="x")
        self.seg_method = tk.StringVar(value="color")
        for label, key in (("Trừ ẢNH NỀN", "background"), ("Theo ĐỘ SÁNG", "brightness"), ("Theo MÀU vải", "color")):
            ttk.Radiobutton(box, text=label, variable=self.seg_method, value=key,
                            command=self._seg_changed).pack(anchor="w")
        ttk.Button(box, text="Bấm ảnh để lấy màu", command=lambda: setattr(self, "click_mode", "color")).pack(fill="x", pady=4)
        self.seg = {k: v for k, v in (
            ("blur", tk.IntVar(value=9)), ("threshold", tk.IntVar(value=35)),
            ("tol_l", tk.DoubleVar(value=27)), ("tol_ab", tk.DoubleVar(value=10)),
            ("morph_kernel", tk.IntVar(value=7)), ("min_area_percent", tk.DoubleVar(value=3)))}
        for label, key, a, b, step in (
            ("Làm mượt", "blur", 1, 31, 2), ("Ngưỡng sáng / trừ nền", "threshold", 0, 255, 1),
            ("Dung sai độ sáng L", "tol_l", 1, 80, 1), ("Dung sai màu a,b", "tol_ab", 1, 60, 1),
            ("Khử tạp chí / nhiễu", "morph_kernel", 1, 31, 2),
            ("Diện tích tối thiểu %", "min_area_percent", .1, 20, .1)):
            self._scale(box, label, self.seg[key], a, b, step, self._seg_changed)
        self.invert = tk.BooleanVar()
        ttk.Checkbutton(box, text="Nền sáng / vật tối (đảo ngưỡng)", variable=self.invert,
                        command=self._seg_changed).pack(anchor="w")
        action = ttk.Labelframe(side, text="Lấy mẫu", padding=8)
        action.pack(fill="x", pady=8)
        ttk.Button(action, text="Chạy tách", command=self._preview_segment).pack(fill="x")
        ttk.Button(action, text="Lấy mẫu vào mảnh đang chọn", command=self._learn_piece).pack(fill="x", pady=4)
        self.piece_state = tk.StringVar()
        ttk.Label(action, textvariable=self.piece_state, wraplength=295).pack(anchor="w")

    def _b2_ui(self):
        right = ttk.Frame(self.b2, width=300, padding=10)
        right.pack(side="right", fill="y")
        self.layout_view = ImageView(self.b2, bg="#dce3e8")
        self.layout_view.pack(fill="both", expand=True, padx=8, pady=8)
        self.layout_view.bind("<Button-1>", self._layout_press)
        self.layout_view.bind("<B1-Motion>", self._layout_drag)
        box = ttk.Labelframe(right, text="Mảnh đang chọn", padding=8)
        box.pack(fill="x")
        for label, key in (("Mảnh 1 · DƯỚI", "bottom"), ("Mảnh 2 · TRÊN (gắp)", "top")):
            ttk.Radiobutton(box, text=label, variable=self.selected_piece, value=key,
                            command=self._piece_to_ui).pack(anchor="w")
        self.angle = tk.DoubleVar()
        self._scale(right, "Góc xoay (độ)", self.angle, -180, 180, .5, self._angle_changed)
        self.relation = tk.StringVar()
        box = ttk.Labelframe(right, text="Quan hệ thiết kế", padding=8)
        box.pack(fill="x", pady=8)
        ttk.Label(box, textvariable=self.relation, font=("Consolas", 11), foreground="#25ddb5").pack(anchor="w")
        ttk.Button(right, text="Căn lại bố cục", command=self._fit_layout).pack(fill="x")

    def _b3_ui(self):
        frame = ttk.Frame(self.b3, padding=8)
        frame.pack(fill="both", expand=True)
        left, right = ttk.Labelframe(frame, text="Mảnh 2 · TRÊN (gắp)"), ttk.Labelframe(frame, text="Mảnh 1 · DƯỚI")
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.tcp_top, self.tcp_bottom = ImageView(left), ImageView(right)
        self.tcp_top.pack(fill="both", expand=True)
        self.tcp_bottom.pack(fill="both", expand=True)
        self.tcp_top.bind("<Button-1>", lambda e: self._tcp_click("top", self.tcp_top, e))
        self.tcp_bottom.bind("<Button-1>", lambda e: self._tcp_click("bottom", self.tcp_bottom, e))
        self.tcp_text = {"top": tk.StringVar(), "bottom": tk.StringVar()}
        ttk.Label(left, textvariable=self.tcp_text["top"], font=("Consolas", 10)).pack(anchor="w")
        ttk.Label(right, textvariable=self.tcp_text["bottom"], font=("Consolas", 10)).pack(anchor="w")
        ttk.Button(self.b3, text="LƯU MẪU THIẾT KẾ", command=self._save).pack(anchor="e", padx=8, pady=5)

    def _camera_ui(self):
        side = ttk.Frame(self.camtab, width=330, padding=10)
        side.pack(side="left", fill="y")
        self.camera_view = ImageView(self.camtab)
        self.camera_view.pack(fill="both", expand=True, padx=8, pady=8)
        box = ttk.Labelframe(side, text="Kết nối camera Basler", padding=8)
        box.pack(fill="x")
        self.device = tk.StringVar()
        self.device_combo = ttk.Combobox(box, textvariable=self.device, state="readonly")
        self.device_combo.pack(fill="x")
        ttk.Button(box, text="Quét Basler", style="Blue.TButton", command=self._scan).pack(fill="x", pady=4)
        ttk.Button(box, text="Kết nối", command=self._connect).pack(fill="x")
        self.camera_status = tk.StringVar(value="Chưa kết nối")
        ttk.Label(box, textvariable=self.camera_status, wraplength=290).pack(anchor="w", pady=4)
        box = ttk.Labelframe(side, text="Thông số chụp", padding=8)
        box.pack(fill="x", pady=8)
        self.exposure, self.gain = tk.DoubleVar(value=10000), tk.DoubleVar(value=0)
        self.camera_width, self.camera_height = tk.IntVar(value=0), tk.IntVar(value=0)
        self.camera_fps = tk.DoubleVar(value=20)
        self.trigger_mode = tk.StringVar(value="Off")
        self._entry(box, "Exposure (µs)", self.exposure)
        self._entry(box, "Gain (dB)", self.gain)
        self._entry(box, "ROI Width (0 = giữ nguyên)", self.camera_width)
        self._entry(box, "ROI Height (0 = giữ nguyên)", self.camera_height)
        self._entry(box, "Giới hạn FPS", self.camera_fps)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Trigger", width=25).pack(side="left")
        ttk.Combobox(
            row, textvariable=self.trigger_mode, values=("Off", "Software", "Line1"), state="readonly", width=12
        ).pack(side="left")
        ttk.Button(box, text="Áp dụng", command=self._apply_camera).pack(fill="x", pady=4)
        ttk.Button(side, text="LIVE / DỪNG · LatestImageOnly", command=self._toggle_camera).pack(fill="x")
        ttk.Button(side, text="PHÁT SOFTWARE TRIGGER", style="Blue.TButton", command=self._software_trigger).pack(
            fill="x", pady=4
        )
        self.cam_overlay = tk.BooleanVar()
        ttk.Checkbutton(side, text="Ảnh gốc ⇄ Overlay phát hiện", variable=self.cam_overlay).pack(anchor="w")
        ttk.Button(side, text="Chụp → CÀI ĐẶT", command=self._camera_design).pack(fill="x", pady=(12, 3))
        ttk.Button(side, text="Chụp → VẬN HÀNH", style="Danger.TButton", command=self._camera_run).pack(fill="x")

    def _run_ui(self):
        side = ttk.Frame(self.run, width=320, padding=10)
        side.pack(side="left", fill="y")
        center = ttk.Frame(self.run, padding=5)
        center.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(self.run, width=340, padding=8)
        right.pack(side="right", fill="y")
        self.run_view = ImageView(center)
        self.run_view.pack(fill="both", expand=True)
        box = ttk.Labelframe(side, text="Nguồn ảnh", padding=8)
        box.pack(fill="x")
        ttk.Button(box, text="Mở ảnh hiện trường…", style="Blue.TButton", command=self._open_scene).pack(fill="x")
        ttk.Button(box, text="CHẠY — TÍNH GẮP & ĐẶT", command=self._detect).pack(fill="x", pady=4)
        self.ghost = tk.BooleanVar()
        ttk.Checkbutton(box, text="Xem trước ĐẶT vải (ghost)", variable=self.ghost, command=self._detect).pack(anchor="w")
        box = ttk.Labelframe(side, text="Cầu nối Robot", padding=8)
        box.pack(fill="x", pady=8)
        self.bridge_button = ttk.Button(box, text="MỞ CẦU NỐI", command=self._toggle_bridge)
        self.bridge_button.pack(fill="x")
        self.bridge_status = tk.StringVar(value="Chưa mở")
        self.bridge_stats = tk.StringVar(value="Khách: -\nKhung: 0\nNhịp: 0 fps")
        ttk.Label(box, textvariable=self.bridge_status, wraplength=285).pack(anchor="w")
        ttk.Label(box, textvariable=self.bridge_stats, font=("Consolas", 10)).pack(anchor="w")
        ttk.Button(side, text="Lưu cảnh (làm mẫu train)", style="Blue.TButton", command=self._capture).pack(fill="x")
        ttk.Button(side, text="Xuất lệnh Robot (.txt)", style="Danger.TButton", command=self._export).pack(fill="x", pady=4)
        box = ttk.Labelframe(right, text="Kết quả", padding=6)
        box.pack(fill="x")
        self.result = tk.StringVar(value="Chưa chạy")
        ttk.Label(box, textvariable=self.result, foreground="#23dfb1", wraplength=315).pack(anchor="w")
        box = ttk.Labelframe(right, text="Lệnh gửi ROBOT", padding=5)
        box.pack(fill="both", expand=True, pady=7)
        self.command = tk.Text(box, width=41, bg="#11161b", fg="white", relief="flat", font=("Consolas", 10))
        self.command.pack(fill="both", expand=True)
        box = ttk.Labelframe(right, text="Nhật ký", padding=5)
        box.pack(fill="x")
        self.log = tk.Text(box, height=8, bg="#11161b", fg="#b9f6ca", relief="flat", font=("Consolas", 9))
        self.log.pack(fill="x")

    def _station_ui(self):
        outer = ttk.Frame(self.station, padding=18)
        outer.pack(fill="both", expand=True)
        box = ttk.Labelframe(outer, text="Hằng số trạm", padding=12)
        box.pack(fill="x")
        self.stvars = {k: v for k, v in (
            ("port", tk.IntVar(value=6001)), ("allow_lan", tk.BooleanVar(value=True)),
            ("angle_sign", tk.DoubleVar(value=-1)), ("angle_offset", tk.DoubleVar(value=0)),
            ("mm_per_pixel", tk.DoubleVar(value=.25)), ("origin_u", tk.DoubleVar()),
            ("origin_v", tk.DoubleVar()), ("max_result_age_ms", tk.IntVar(value=1500)),
            ("scale_tolerance_percent", tk.DoubleVar(value=25)))}
        for label, key in (("Cổng cầu nối", "port"), ("Góc trục W — dấu", "angle_sign"),
                           ("Bù góc (°)", "angle_offset"), ("mm / pixel", "mm_per_pixel"),
                           ("Gốc U", "origin_u"), ("Gốc V", "origin_v"),
                           ("Tuổi kết quả tối đa (ms)", "max_result_age_ms")):
            self._entry(box, label, self.stvars[key])
        ttk.Checkbutton(box, text="Nhận ảnh từ mạng LAN", variable=self.stvars["allow_lan"]).pack(anchor="w")
        box = ttk.Labelframe(outer, text="Tinh chỉnh khi chạy", padding=12)
        box.pack(fill="x", pady=14)
        self.algorithm = tk.StringVar(value="auto")
        self.algo_combo = ttk.Combobox(box, state="readonly", width=43, values=list(ALGORITHMS.values()))
        self.algo_combo.pack(anchor="w")
        self.algo_combo.bind("<<ComboboxSelected>>", lambda _e: self.algorithm.set(
            next(k for k, v in ALGORITHMS.items() if v == self.algo_combo.get())))
        self._entry(box, "Dung sai tỉ lệ ± %", self.stvars["scale_tolerance_percent"])
        box = ttk.Labelframe(outer, text="Hiệu chuẩn mặt phẳng · Pixel → Robot", padding=12)
        box.pack(fill="x")
        ttk.Label(box, text="Mỗi dòng: u, v, X, Y · cần ít nhất 4 điểm không thẳng hàng").pack(anchor="w")
        self.calibration_points = tk.Text(
            box, height=5, width=58, bg="#11161b", fg="#e8edf2", relief="flat", font=("Consolas", 10)
        )
        self.calibration_points.pack(side="left", fill="x", expand=True, pady=5)
        cal_actions = ttk.Frame(box)
        cal_actions.pack(side="left", fill="y", padx=(10, 0))
        ttk.Button(cal_actions, text="TÍNH HOMOGRAPHY", command=self._calibrate).pack(fill="x")
        ttk.Button(cal_actions, text="XÓA HOMOGRAPHY", style="Danger.TButton", command=self._clear_calibration).pack(
            fill="x", pady=4
        )
        self.calibration_status = tk.StringVar(value="Chưa hiệu chuẩn · đang dùng mm/pixel")
        ttk.Label(cal_actions, textvariable=self.calibration_status, wraplength=330).pack(anchor="w")
        ttk.Button(outer, text="LƯU CẤU HÌNH TRẠM", command=self._save_station).pack(anchor="e")

    def _learn_ui(self):
        left = ttk.Frame(self.learn, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self.learn, padding=8)
        right.pack(fill="both", expand=True)
        self.lessons = tk.Listbox(left, width=37, bg="#11161b", fg="white", selectbackground="#168bd1")
        names = ["Pixel, BGR, Gray và LAB", "Histogram & ánh sáng", "Threshold và mask", "Morphology",
                 "Contour và minAreaRect", "Tâm, góc và TCP", "Trừ ảnh nền", "Template matching ảnh xám",
                 "ORB và homography", "Che khuất và biến dạng", "Pixel → mm", "Quan hệ PICK/PLACE",
                 "Basler exposure/gain", "TCP bridge và frame age", "An toàn Robot"]
        for i, name in enumerate(names, 1):
            self.lessons.insert("end", f"{i:02d} · {name}")
        self.lessons.pack(fill="y", expand=True)
        self.lessons.bind("<<ListboxSelect>>", self._lesson)
        self.lesson_text = tk.Text(right, height=8, bg="#151a20", fg="#dce8f3", relief="flat",
                                   font=("Segoe UI", 11), wrap="word")
        self.lesson_text.pack(fill="x")
        self.lesson_view = ImageView(right)
        self.lesson_view.pack(fill="both", expand=True, pady=(8, 0))

    @staticmethod
    def _scale(parent, label, var, a, b, step, callback):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label).pack(anchor="w")
        tk.Scale(row, from_=a, to=b, resolution=step, variable=var, orient="horizontal",
                 bg="#20252b", fg="#f6a919", troughcolor="#46515d", highlightthickness=0,
                 command=lambda _v: callback()).pack(fill="x")

    @staticmethod
    def _entry(parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=25).pack(side="left")
        ttk.Entry(row, textvariable=var, width=14).pack(side="left")

    def _asset(self, filename):
        return self.repo.load_image(self.product, filename)

    def _piece(self):
        return self.product.piece(self.selected_piece.get())

    def _new_product(self):
        name = simpledialog.askstring("Thêm sản phẩm", "Tên sản phẩm mới\n(mẫu hiện tại sẽ được lưu thành bản sao):")
        if name:
            try:
                self.product = self.repo.create(name, self.product)
                self._product_changed()
            except Exception as exc:
                messagebox.showerror("Vision Lab", str(exc))

    def _import_hoxco(self):
        folder = filedialog.askdirectory(title="Chọn thư mục sản phẩm HOXCO có template.json")
        if not folder:
            return
        name = simpledialog.askstring("Nhập sản phẩm HOXCO", "Tên sản phẩm mới:", initialvalue=f"{Path(folder).name}-import")
        if not name:
            return
        try:
            self.product = self.repo.import_hoxco(folder, name)
            self._product_changed()
            self._log("Đã nhập bản sao sản phẩm HOXCO; dữ liệu gốc không bị thay đổi.")
        except Exception as exc:
            messagebox.showerror("Nhập HOXCO", str(exc))

    def _delete_product(self):
        if len(self.repo.names()) <= 1:
            return messagebox.showinfo("Vision Lab", "Phải giữ ít nhất một sản phẩm.")
        if messagebox.askyesno("Xóa sản phẩm", f"Xóa '{self.product.name}'? Không thể khôi phục."):
            self.repo.delete(self.product.name)
            self.product = self.repo.load(self.repo.names()[0])
            self._product_changed()

    def _switch_product(self):
        if self.product_var.get() != self.product.name:
            self.product = self.repo.load(self.product_var.get())
            self._product_changed()

    def _product_changed(self):
        self.engine.update_product(self.product, self._asset)
        self.image = self.overlay = self._asset(self.product.source_image_file)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(
            json.dumps({"last_product": self.product.name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._load_ui()

    def _load_ui(self):
        self.products["values"] = self.repo.names()
        self.product_var.set(self.product.name)
        self.design_view.show(self.image)
        self.run_view.show(self.overlay)
        self._piece_to_ui()
        self._layout_render()
        self._tcp_render()
        for key, var in self.stvars.items():
            var.set(getattr(self.product.station, key))
        self.algorithm.set(self.product.station.algorithm)
        self.algo_combo.set(ALGORITHMS[self.product.station.algorithm])
        if hasattr(self, "calibration_status"):
            status = "Homography 3×3 đã sẵn sàng" if self.product.station.homography else \
                "Chưa hiệu chuẩn · đang dùng mm/pixel"
            self.calibration_status.set(status)

    def _piece_to_ui(self):
        p = self._piece()
        self.seg_method.set(p.params.get("method", "color"))
        for key, var in self.seg.items():
            var.set(p.params.get(key, var.get()))
        self.invert.set(p.params.get("invert", False))
        self.angle.set(p.design_angle)
        self.piece_state.set("\n".join(f"{x.name}: {'đã có mẫu' if x.template_file else 'chưa lấy mẫu'}"
                                         for x in self.product.pieces))

    def _seg_changed(self):
        p = self._piece()
        p.params["method"] = self.seg_method.get()
        p.params["invert"] = self.invert.get()
        for key, var in self.seg.items():
            p.params[key] = var.get()

    def _open_design(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image = open_image(path)
            self.product.source_image_file = self.repo.save_image(self.product, "design_source.jpg", self.image)
            self.repo.save(self.product)
            self.design_view.show(self.image)

    def _set_background(self):
        if self.image is not None:
            self.product.background_file = self.repo.save_image(self.product, "background.png", self.image)
            self.repo.save(self.product)
            self._log("Đã lấy ảnh nền.")

    def _design_click(self, event):
        point = self.design_view.point(event)
        if point and self.image is not None and self.click_mode == "color":
            color = self.engine.sample_color(self.image, *point, self._piece())
            self.click_mode = None
            self._log(f"Màu {self._piece().name}: LAB {np.round(color).astype(int).tolist()}")
            self._preview_segment()

    def _preview_segment(self):
        if self.image is None:
            return
        self._seg_changed()
        mask, contours = self.engine.segment_piece(self.image, self._piece())
        shown = self.image.copy()
        cv2.drawContours(shown, contours, -1, tuple(self._piece().draw_color_bgr), 3)
        self.design_view.show(shown)
        self.lesson_view.show(mask)

    def _learn_piece(self):
        if self.image is None:
            return
        try:
            self._seg_changed()
            p = self._piece()
            sprite, mask, _ = self.engine.learn_piece(self.image, p)
            p.template_file = self.repo.save_image(self.product, f"{p.key}_template.png", sprite)
            p.mask_file = self.repo.save_image(self.product, f"{p.key}_mask.png", mask)
            self.repo.save(self.product)
            self._piece_to_ui()
            self._layout_render()
            self._tcp_render()
        except Exception as exc:
            messagebox.showerror("Vision Lab", str(exc))

    def _fit_layout(self):
        b, t = self.product.piece("bottom"), self.product.piece("top")
        b.design_x, b.design_y, t.design_x, t.design_y = 0, 110, 0, -110
        self._layout_render()

    def _layout_render(self):
        canvas = np.full((760, 1150, 3), (224, 231, 236), np.uint8)
        for p in self.product.pieces:
            sprite, mask = self._asset(p.template_file), self._asset(p.mask_file)
            if sprite is None or mask is None:
                continue
            sprite, mask = self._rotate_sprite(sprite, mask, p.design_angle)
            x, y = round(575 + p.design_x - sprite.shape[1] / 2), round(380 + p.design_y - sprite.shape[0] / 2)
            self._paste(canvas, sprite, mask, x, y)
            label = "MANH TREN (GAP)" if p.is_top else "MANH DUOI"
            cv2.putText(canvas, f"{label} {p.design_angle:+.1f} deg", (max(4, x), max(24, y)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, tuple(p.draw_color_bgr), 2)
        self.layout_view.show(canvas)
        b, t = self.product.piece("bottom"), self.product.piece("top")
        self.relation.set(f"Δx = {t.design_x-b.design_x:+.0f}px\nΔy = {t.design_y-b.design_y:+.0f}px\n"
                          f"Δθ = {t.design_angle-b.design_angle:+.1f}°")

    def _layout_press(self, event):
        self.drag_anchor = self.layout_view.point(event)

    def _layout_drag(self, event):
        point = self.layout_view.point(event)
        if point and getattr(self, "drag_anchor", None):
            self._piece().design_x += point[0] - self.drag_anchor[0]
            self._piece().design_y += point[1] - self.drag_anchor[1]
            self.drag_anchor = point
            self._layout_render()

    def _angle_changed(self):
        self._piece().design_angle = self.angle.get()
        self._layout_render()

    def _tcp_render(self):
        for key, view in (("top", self.tcp_top), ("bottom", self.tcp_bottom)):
            p, image = self.product.piece(key), self._asset(self.product.piece(key).template_file)
            if image is not None:
                shown = image.copy()
                point = (round(image.shape[1]/2+p.tcp_offset_local[0]), round(image.shape[0]/2+p.tcp_offset_local[1]))
                cv2.drawMarker(shown, point, (20, 20, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(shown, "TCP", (point[0]+7, point[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .6, (20,20,255), 2)
                view.show(shown)
            self.tcp_text[key].set(f"TCP: ({p.tcp_offset_local[0]:+.0f}, {p.tcp_offset_local[1]:+.0f}) px so với tâm")

    def _tcp_click(self, key, view, event):
        point, image = view.point(event), self._asset(self.product.piece(key).template_file)
        if point and image is not None:
            self.product.piece(key).tcp_offset_local = [point[0]-image.shape[1]/2, point[1]-image.shape[0]/2]
            self._tcp_render()

    def _save(self):
        self.repo.save(self.product)
        self._log("Đã lưu mẫu thiết kế.")

    def _scan(self):
        try:
            if not self.camera.available:
                self.camera_status.set("Chưa cài pypylon/Basler pylon Runtime")
                return
            devices = self.camera.enumerate()
            self.device_combo["values"] = [f"{d['model']} | {d['serial']} | {d['ip']}" for d in devices]
            if devices:
                self.device_combo.current(0)
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
                self.exposure.get(),
                self.gain.get(),
                self.camera_width.get(),
                self.camera_height.get(),
                self.camera_fps.get(),
                self.trigger_mode.get(),
            )
            self.camera_status.set(
                f"{values.get('Width', 0)}×{values.get('Height', 0)} · "
                f"Exposure={values.get('ExposureTime',0):.0f}µs · Gain={values.get('Gain',0):.1f}dB · "
                f"Trigger={values.get('TriggerMode', 'Off')}"
            )
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _software_trigger(self):
        try:
            self.camera.software_trigger()
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _toggle_camera(self):
        try:
            self.camera.stop() if self.camera.running else self.camera.start()
        except Exception as exc:
            self.camera_status.set(str(exc))

    def _camera_frame(self, frame, fps):
        self._latest(("camera", frame, fps))

    def _camera_design(self):
        if self.image is not None:
            self.product.source_image_file = self.repo.save_image(self.product, "design_source.jpg", self.image)
            self.repo.save(self.product)
            self.design_view.show(self.image)
            self.tabs.select(self.design)

    def _camera_run(self):
        self.run_view.show(self.image)
        self.tabs.select(self.run)

    def _open_scene(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image = open_image(path)
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
        objects = {d.type_id: d for d in self.detections}
        pick, place = objects.get(1), objects.get(0)
        if not pick or not place:
            self.result.set(f"✕ Không định vị đủ hai mảnh\nXử lý: {elapsed:.1f} ms")
            text = "// Không khớp đủ 2 mảnh."
        else:
            st = self.product.station
            pxy = self._pixel_to_robot(pick.tcp_u, pick.tcp_v)
            qxy = self._pixel_to_robot(place.tcp_u, place.tcp_v)
            pw, qw = st.angle_sign*pick.angle+st.angle_offset, st.angle_sign*place.angle+st.angle_offset
            self.result.set(f"✓ Khớp mẫu thành công\nTRÊN: ({pick.tcp_u:.0f},{pick.tcp_v:.0f}) {pick.angle:+.1f}° · {pick.score:.2f}\n"
                            f"DƯỚI: ({place.tcp_u:.0f},{place.tcp_v:.0f}) {place.angle:+.1f}° · {place.score:.2f}\n"
                            f"Xử lý: {elapsed:.1f} ms")
            text = (f"// ===== VISION LAB -> ROBOT =====\n\nPICK:\n  u,v = {pick.tcp_u:.0f}, {pick.tcp_v:.0f}px\n"
                    f"  theta = {pick.angle:+.2f}°\n  X,Y = {pxy[0]:.1f}, {pxy[1]:.1f}mm\n  W = {pw:+.2f}°\n\n"
                    f"PLACE:\n  u,v = {place.tcp_u:.0f}, {place.tcp_v:.0f}px\n  theta = {place.angle:+.2f}°\n"
                    f"  X,Y = {qxy[0]:.1f}, {qxy[1]:.1f}mm\n  W = {qw:+.2f}°\n\nROTATE = {qw-pw:+.2f}°")
        self.command.delete("1.0", "end")
        self.command.insert("1.0", text)
        for note in notes:
            self._log(note)

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
            self.bridge_button.configure(text="MỞ CẦU NỐI")
        else:
            self._station_to_model()
            host = "0.0.0.0" if self.product.station.allow_lan else "127.0.0.1"
            self.bridge.start(host, self.product.station.port)
            self.bridge_button.configure(text="ĐÓNG CẦU NỐI", style="Danger.TButton")

    def _capture(self):
        if self.image is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = self.repo.capture_folder(self.product)
        self.repo.save_image(self.product, f"captures/scene_{stamp}.jpg", self.image)
        (folder/f"scene_{stamp}.json").write_text(json.dumps({"time":stamp,"algorithm":self.product.station.algorithm,
            "objects":[{"type":d.type_id,"tcp":[d.tcp_u,d.tcp_v],"angle":d.angle,"score":d.score} for d in self.detections]}, indent=2))
        self._log(f"Đã lưu cảnh scene_{stamp}.jpg")

    def _export(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt")
        if path:
            Path(path).write_text(self.command.get("1.0", "end"), encoding="utf-8")

    def _station_to_model(self):
        for key, var in self.stvars.items():
            setattr(self.product.station, key, var.get())
        self.product.station.algorithm = self.algorithm.get()

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
                values = [float(part.strip()) for part in raw.replace("=>", ",").replace(";", ",").split(",") if part.strip()]
                if len(values) != 4:
                    raise ValueError(f"Dòng {number} phải có đúng 4 số: u, v, X, Y")
                image_points.append(values[:2])
                robot_points.append(values[2:])
            if len(image_points) < 4:
                raise ValueError("Cần ít nhất 4 cặp điểm hiệu chuẩn")
            matrix, inliers = cv2.findHomography(
                np.asarray(image_points, np.float32),
                np.asarray(robot_points, np.float32),
                cv2.RANSAC,
                1.5,
            )
            if matrix is None:
                raise ValueError("Không tính được homography; kiểm tra các điểm có thẳng hàng không")
            projected = cv2.perspectiveTransform(
                np.asarray(image_points, np.float32).reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((projected - np.asarray(robot_points)) ** 2, axis=1))))
            self.product.station.homography = matrix.reshape(-1).astype(float).tolist()
            self.calibration_status.set(f"Đã hiệu chuẩn {len(image_points)} điểm · RMS {rms:.3f} mm")
            self._log(self.calibration_status.get())
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

    def _lesson(self, _e):
        index = self.lessons.curselection()
        if not index:
            return
        text = [
            "LAB tách độ sáng L khỏi màu a,b, phù hợp hơn BGR khi ánh sáng dao động.",
            "Exposure quá cao làm mất texture; quá thấp tăng nhiễu và mất điểm đặc trưng.",
            "Mask tốt có vật trắng, nền đen, ít lỗ và ít vùng giả.",
            "Opening bỏ nhiễu; Closing lấp lỗ. Kernel quá lớn sẽ làm biến dạng biên.",
            "Contour mô tả biên; minAreaRect trả tâm, kích thước và góc cạnh dài.",
            "TCP là điểm hút. Offset TCP phải xoay theo pose vật.",
            "Trừ nền mạnh khi camera và ánh sáng cố định; phải lấy nền lại khi trạm đổi.",
            "Ảnh xám/edge matching dễ hiểu nhưng phải quét góc và scale.",
            "ORB + homography chịu xoay, scale và che khuất khi vật có hoa văn.",
            "Vải biến dạng cần kết hợp màu, contour và feature thay vì một tiêu chí.",
            "mm/pixel là mô hình cơ bản; homography cần bốn điểm chuẩn trở lên.",
            "Δx, Δy, Δθ lưu quan hệ thiết kế giữa mảnh gắp và mảnh đặt.",
            "Basler cần tắt auto trước khi đặt Exposure/Gain; LatestImageOnly chống frame cũ.",
            "Bridge chỉ nên giữ frame mới nhất và từ chối kết quả quá tuổi.",
            "Thử Robot với Z an toàn, tốc độ thấp và đầu hút tắt trước.",
        ][index[0]]
        self.lesson_text.delete("1.0", "end")
        self.lesson_text.insert("1.0", text)
        self.lesson_view.show(self.overlay if self.overlay is not None else self.image)

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
                    self.image = frame
                    shown = frame
                    if self.cam_overlay.get():
                        self.detections, shown, _ = self.engine.process(frame, self.product.station.algorithm)
                    self.camera_view.show(shown)
                    self.camera_status.set(f"LIVE · {fps:.1f}fps · frame mới nhất")
                else:
                    _, self.overlay, self.detections, stats = event
                    with self._bridge_lock:
                        if self._bridge_raw is not None:
                            self.image = self._bridge_raw.copy()
                    if self.ghost.get():
                        self.overlay = self._ghost_overlay(self.overlay, self.detections)
                    self.run_view.show(self.overlay)
                    self.bridge_stats.set(f"Khách: {stats.client}\nKhung: {stats.frames}\nNhịp: {stats.fps:.1f}fps\n"
                                          f"Xử lý: {stats.processing_ms:.1f}ms")
                    notes = [self._bridge_note] if self._bridge_note else []
                    self._show_result(stats.processing_ms, notes)
        except queue.Empty:
            pass
        try:
            while True:
                event = self.messages.get_nowait()
                if event[0] == "camera":
                    self.camera_status.set(event[1])
                else:
                    _, text, stats = event
                    self.bridge_status.set(text)
                    self._log(text)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _log(self, text):
        if hasattr(self, "log"):
            self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n")
            self.log.see("end")

    def _ghost_overlay(self, image, detections):
        bottom_detection = next((d for d in detections if d.type_id == 0), None)
        top_piece = self.product.piece("top")
        bottom_piece = self.product.piece("bottom")
        sprite, mask = self._asset(top_piece.template_file), self._asset(top_piece.mask_file)
        if bottom_detection is None or sprite is None or mask is None:
            return image
        design_delta = (
            top_piece.design_x - bottom_piece.design_x,
            top_piece.design_y - bottom_piece.design_y,
        )
        radians = np.deg2rad(bottom_detection.angle - bottom_piece.design_angle)
        dx = design_delta[0] * np.cos(radians) - design_delta[1] * np.sin(radians)
        dy = design_delta[0] * np.sin(radians) + design_delta[1] * np.cos(radians)
        target_angle = bottom_detection.angle + top_piece.design_angle - bottom_piece.design_angle
        sprite, mask = self._rotate_sprite(sprite[:, :, :3], mask, target_angle)
        x = round(bottom_detection.center_u + dx - sprite.shape[1] / 2)
        y = round(bottom_detection.center_v + dy - sprite.shape[0] / 2)
        composed = image.copy()
        self._paste(composed, sprite, mask, x, y)
        ghost = cv2.addWeighted(image, .55, composed, .45, 0)
        cv2.putText(
            ghost,
            "VI TRI DAT THEO THIET KE",
            (max(4, x), max(24, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .58,
            (30, 50, 255),
            2,
            cv2.LINE_AA,
        )
        return ghost

    @staticmethod
    def _rotate_sprite(sprite, mask, angle):
        h, w = sprite.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        c, s = abs(matrix[0,0]), abs(matrix[0,1])
        nw, nh = int(h*s+w*c), int(h*c+w*s)
        matrix[0,2] += nw/2-w/2
        matrix[1,2] += nh/2-h/2
        return cv2.warpAffine(sprite, matrix, (nw,nh)), cv2.warpAffine(mask, matrix, (nw,nh))

    @staticmethod
    def _paste(canvas, sprite, mask, x, y):
        h,w = sprite.shape[:2]
        x0,y0,x1,y1 = max(0,x),max(0,y),min(canvas.shape[1],x+w),min(canvas.shape[0],y+h)
        if x0>=x1 or y0>=y1: return
        sx,sy=x0-x,y0-y
        crop=sprite[sy:sy+y1-y0,sx:sx+x1-x0]
        m=(mask[sy:sy+y1-y0,sx:sx+x1-x0].astype(float)/255)[:,:,None]
        canvas[y0:y1,x0:x1]=(canvas[y0:y1,x0:x1]*(1-m)+crop*m).astype(np.uint8)

    def _close(self):
        self.camera.close()
        self.bridge.stop()
        self.repo.save(self.product)
        self.destroy()


if __name__ == "__main__":
    VisionLab().mainloop()
