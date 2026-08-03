from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SEGMENT_PARAMS = {
    "method": "color",
    "blur": 9,
    "threshold": 35,
    "bg_threshold": 20,
    "invert": False,
    "morph_kernel": 7,
    "morph_iterations": 2,
    "min_area_percent": 3.0,
    "color_tolerance": 15.0,
    "merge_fragments": False,
    "auto_threshold": False,
    "pattern_threshold": 30,
}


@dataclass
class PieceModel:
    key: str
    name: str
    type_id: int
    is_top: bool
    draw_color_bgr: list[int]
    params: dict = field(default_factory=lambda: dict(DEFAULT_SEGMENT_PARAMS))
    reference_lab: list[float] | None = None
    tcp_offset_local: list[float] = field(default_factory=lambda: [0.0, 0.0])
    design_x: float = 0.0
    design_y: float = 0.0
    design_angle: float = 0.0
    template_file: str | None = None
    mask_file: str | None = None


@dataclass
class StationConfig:
    port: int = 6001
    allow_lan: bool = True
    angle_sign: float = -1.0
    angle_offset: float = 0.0
    mm_per_pixel: float = 0.25
    origin_u: float = 0.0
    origin_v: float = 0.0
    algorithm: str = "auto"
    scale_tolerance_percent: float = 25.0
    max_result_age_ms: int = 1500
    homography: list[float] | None = None


@dataclass
class ProductModel:
    name: str
    pieces: list[PieceModel] = field(
        default_factory=lambda: [
            PieceModel("bottom", "Mảnh 1 · DƯỚI", 0, False, [70, 230, 90]),
            PieceModel("top", "Mảnh 2 · TRÊN (gắp)", 1, True, [255, 180, 30]),
        ]
    )
    station: StationConfig = field(default_factory=StationConfig)
    source_image_file: str | None = None
    background_file: str | None = None

    def piece(self, key: str) -> PieceModel:
        for piece in self.pieces:
            if piece.key == key:
                return piece
        raise KeyError(key)


class ProductRepository:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_name(name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name.strip())
        return safe[:80] or "san-pham-moi"

    def names(self) -> list[str]:
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "product.json").exists()
        )

    def create(self, name: str, clone: ProductModel | None = None) -> ProductModel:
        name = self.safe_name(name)
        folder = self.root / name
        if folder.exists():
            raise FileExistsError(f"Sản phẩm '{name}' đã tồn tại")
        folder.mkdir(parents=True)
        if clone is None:
            product = ProductModel(name=name)
        else:
            payload = json.loads(json.dumps(asdict(clone)))
            product = self._from_payload(payload)
            product.name = name
            source_folder = self.root / clone.name
            for filename in self._asset_names(clone):
                source = source_folder / filename
                if source.exists():
                    shutil.copy2(source, folder / filename)
        self.save(product)
        return product

    def import_hoxco(self, source: str | Path, name: str | None = None) -> ProductModel:
        """Import a user-owned HOXCO product folder without modifying the source."""
        source = Path(source)
        config_file = source / "template.json"
        if not config_file.exists():
            raise FileNotFoundError("Thư mục không có template.json của HOXCO")
        payload = json.loads(config_file.read_text(encoding="utf-8"))
        if not {"p1", "p2", "top"}.issubset(payload):
            raise ValueError("template.json không đúng định dạng sản phẩm HOXCO")
        product = self.create(name or f"{source.name}-import")
        top_key = payload["top"]
        bottom_key = "p1" if top_key == "p2" else "p2"
        mapping = {"top": top_key, "bottom": bottom_key}
        for key, old_key in mapping.items():
            piece = product.piece(key)
            old = payload[old_key]
            params = old.get("params", {})
            piece.reference_lab = old.get("med_lab") or [
                params.get("ref_l", 0), params.get("ref_a", 128), params.get("ref_b", 128)
            ]
            piece.tcp_offset_local = list(old.get("tcp", [0, 0]))
            piece.params.update({
                "method": params.get("method", "color"),
                "blur": max(1, int(params.get("blur", 9))),
                "threshold": int(params.get("thresh_value", 35)),
                "bg_threshold": int(params.get("bg_tol", 20)),
                "invert": bool(params.get("invert", False)),
                "morph_kernel": max(1, int(params.get("morph_kernel", 7))),
                "morph_iterations": max(0, int(params.get("morph_iter", 2))),
                "min_area_percent": float(params.get("min_area_pct", 3)),
                "tol_l": float(params.get("tol_l", 27)),
                "tol_ab": float(params.get("tol_ab", 10)),
                "merge_fragments": bool(params.get("merge_fragments", False)),
                "auto_threshold": bool(params.get("auto_threshold", False)),
                "pattern_threshold": int(params.get("pattern_var_thresh", 30)),
            })
            sprite_path = source / f"{old_key}.png"
            sprite = cv2.imdecode(np.fromfile(str(sprite_path), np.uint8), cv2.IMREAD_UNCHANGED)
            if sprite is None:
                raise FileNotFoundError(f"Thiếu {old_key}.png")
            piece.template_file = self.save_image(product, f"{key}_template.png", sprite[:, :, :3])
            alpha = sprite[:, :, 3] if sprite.ndim == 3 and sprite.shape[2] == 4 else np.full(sprite.shape[:2], 255, np.uint8)
            piece.mask_file = self.save_image(product, f"{key}_mask.png", alpha)
        pose1, pose2 = payload.get("pose1", [0, 0, 0]), payload.get("pose2", [0, 0, 0])
        pieces_by_old_key = {
            "p1": product.piece("top" if top_key == "p1" else "bottom"),
            "p2": product.piece("top" if top_key == "p2" else "bottom"),
        }
        for old_key, pose in (("p1", pose1), ("p2", pose2)):
            pieces_by_old_key[old_key].design_x = float(pose[0] - pose1[0])
            pieces_by_old_key[old_key].design_y = float(pose[1] - pose1[1])
            pieces_by_old_key[old_key].design_angle = float(pose[2])
        background = source / "bg.png"
        if background.exists():
            image = cv2.imdecode(np.fromfile(str(background), np.uint8), cv2.IMREAD_COLOR)
            product.background_file = self.save_image(product, "background.png", image)
        captures = sorted((source / "captures").glob("*.jpg")) if (source / "captures").exists() else []
        if captures:
            image = cv2.imdecode(np.fromfile(str(captures[-1]), np.uint8), cv2.IMREAD_COLOR)
            product.source_image_file = self.save_image(product, "design_source.jpg", image)
        self.save(product)
        return product

    def delete(self, name: str) -> None:
        folder = (self.root / self.safe_name(name)).resolve()
        root = self.root.resolve()
        if folder.parent != root or not folder.exists():
            raise ValueError("Đường dẫn sản phẩm không hợp lệ")
        shutil.rmtree(folder)

    def save(self, product: ProductModel) -> None:
        folder = self.root / self.safe_name(product.name)
        folder.mkdir(parents=True, exist_ok=True)
        product.name = folder.name
        (folder / "captures").mkdir(exist_ok=True)
        payload = asdict(product)
        (folder / "product.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, name: str) -> ProductModel:
        folder = self.root / self.safe_name(name)
        payload = json.loads((folder / "product.json").read_text(encoding="utf-8"))
        return self._from_payload(payload)

    def save_image(self, product: ProductModel, filename: str, image: np.ndarray) -> str:
        folder = self.root / self.safe_name(product.name)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(path.suffix or ".png", image)[1].tofile(str(path))
        return path.name

    def load_image(self, product: ProductModel, filename: str | None) -> np.ndarray | None:
        if not filename:
            return None
        path = self.root / self.safe_name(product.name) / filename
        if not path.exists():
            return None
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def capture_folder(self, product: ProductModel) -> Path:
        folder = self.root / self.safe_name(product.name) / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def _from_payload(payload: dict) -> ProductModel:
        pieces = [PieceModel(**piece) for piece in payload.get("pieces", [])]
        station = StationConfig(**payload.get("station", {}))
        return ProductModel(
            name=payload["name"],
            pieces=pieces or ProductModel(name="x").pieces,
            station=station,
            source_image_file=payload.get("source_image_file"),
            background_file=payload.get("background_file"),
        )

    @staticmethod
    def _asset_names(product: ProductModel) -> list[str]:
        names = [product.source_image_file, product.background_file]
        for piece in product.pieces:
            names.extend([piece.template_file, piece.mask_file])
        return [name for name in names if name]
