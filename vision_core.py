from dataclasses import dataclass
import numpy as np


@dataclass
class Detection:
    type_id: int
    class_name: str
    center_u: float
    center_v: float
    tcp_u: float
    tcp_v: float
    width: float
    height: float
    angle: float
    area: float
    score: float
    contour: np.ndarray

    def to_robot_tuple(self):
        return (int(self.type_id), float(self.tcp_u), float(self.tcp_v),
                float(self.width), float(self.height), float(self.angle))
