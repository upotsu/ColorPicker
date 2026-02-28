import sys
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget
)


def clamp_int(v: float, lo: int = 0, hi: int = 255) -> int:
    return int(max(lo, min(hi, round(v))))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_hsv(r: int, g: int, b: int):
    """Return HSV as (H in degrees 0..360, S 0..100, V 0..100)."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    mx = max(rf, gf, bf)
    mn = min(rf, gf, bf)
    diff = mx - mn

    if diff == 0:
        h = 0.0
    elif mx == rf:
        h = (60 * ((gf - bf) / diff) + 360) % 360
    elif mx == gf:
        h = (60 * ((bf - rf) / diff) + 120) % 360
    else:
        h = (60 * ((rf - gf) / diff) + 240) % 360

    s = 0.0 if mx == 0 else (diff / mx)
    v = mx
    return (h, s * 100.0, v * 100.0)


@dataclass
class ColorSample:
    r: int
    g: int
    b: int
    a: int = 255

    @property
    def hex(self) -> str:
        return rgb_to_hex(self.r, self.g, self.b)

    @property
    def hex8(self) -> str:
        # #RRGGBBAA
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}{self.a:02X}"

    @property
    def hsv(self):
        return rgb_to_hsv(self.r, self.g, self.b)


class ImageCanvas(QLabel):
    pixelSampled = Signal(object)   # ColorSample
    rectSampled = Signal(object)    # (ColorSample avg, QRect imgRect, int count)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

        self._img: QImage | None = None
        self._pix: QPixmap | None = None

        self._scale = 1.0
        self._min_scale = 0.05
        self._max_scale = 20.0

        self._dragging = False
        self._drag_start = QPoint()
        self._drag_end = QPoint()

        # Numpy copy (H, W, 4) RGBA
        self._np_rgba: np.ndarray | None = None

        self._last_mouse_pos = QPoint(-1, -1)

    def has_image(self) -> bool:
        return self._img is not None

    def load_image(self, path: str):
        img = QImage(path)
        if img.isNull():
            raise ValueError("画像の読み込みに失敗しました。形式/破損を確認してください。")

        img = img.convertToFormat(QImage.Format_RGBA8888)
        self._img = img
        self._pix = QPixmap.fromImage(img)

        w, h = img.width(), img.height()
        ptr = img.bits()
        bpl = img.bytesPerLine()
        buf = np.frombuffer(ptr, dtype=np.uint8, count=bpl * h)
        arr = buf.reshape((h, bpl))
        rgba = arr[:, : w * 4].reshape((h, w, 4)).copy()
        self._np_rgba = rgba

        self._scale = 1.0
        self._dragging = False
        self._drag_start = QPoint()
        self._drag_end = QPoint()
        self._last_mouse_pos = QPoint(-1, -1)

        self._update_display()

    def set_scale(self, s: float):
        self._scale = max(self._min_scale, min(self._max_scale, float(s)))
        self._update_display()

    def get_scale(self) -> float:
        return self._scale

    def _update_display(self):
        if not self._pix:
            self.clear()
            return
        scaled = self._pix.scaled(
            int(self._pix.width() * self._scale),
            int(self._pix.height() * self._scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.update()

    def _widget_to_image_xy(self, widget_pos: QPoint) -> tuple[int, int] | None:
        if not self._img or not self.pixmap():
            return None

        pm = self.pixmap()
        pm_w, pm_h = pm.width(), pm.height()

        x0 = (self.width() - pm_w) // 2
        y0 = (self.height() - pm_h) // 2

        x = widget_pos.x() - x0
        y = widget_pos.y() - y0

        if x < 0 or y < 0 or x >= pm_w or y >= pm_h:
            return None

        ix = int(x / self._scale)
        iy = int(y / self._scale)

        if ix < 0 or iy < 0 or ix >= self._img.width() or iy >= self._img.height():
            return None
        return ix, iy

    def _sample_pixel(self, ix: int, iy: int) -> ColorSample:
        rgba = self._np_rgba[iy, ix]
        r, g, b, a = int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])
        return ColorSample(r, g, b, a)

    def _sample_rect_avg(self, rect: QRect) -> tuple[ColorSample, int]:
        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(self._img.width() - 1, rect.right())
        y2 = min(self._img.height() - 1, rect.bottom())
        if x2 < x1 or y2 < y1:
            return ColorSample(0, 0, 0, 255), 0

        region = self._np_rgba[y1:y2 + 1, x1:x2 + 1, :4].astype(np.float32)  # RGBA
        count = region.shape[0] * region.shape[1]
        if count <= 0:
            return ColorSample(0, 0, 0, 255), 0

        mean = region.mean(axis=(0, 1))
        r, g, b, a = clamp_int(mean[0]), clamp_int(mean[1]), clamp_int(mean[2]), clamp_int(mean[3])
        return ColorSample(r, g, b, a), count

    def mousePressEvent(self, e):
        if not self.has_image():
            return
        if e.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start = e.position().toPoint()
            self._drag_end = self._drag_start
            self.update()

    def mouseMoveEvent(self, e):
        self._last_mouse_pos = e.position().toPoint()
        if not self.has_image():
            return

        if self._dragging:
            self._drag_end = e.position().toPoint()
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, e):
        if not self.has_image():
            return
        if e.button() != Qt.LeftButton:
            return

        self._dragging = False
        end = e.position().toPoint()

        dx = abs(end.x() - self._drag_start.x())
        dy = abs(end.y() - self._drag_start.y())

        if dx <= 3 and dy <= 3:
            img_xy = self._widget_to_image_xy(end)
            if img_xy is None:
                return
            ix, iy = img_xy
            sample = self._sample_pixel(ix, iy)
            self.pixelSampled.emit(sample)
        else:
            p1 = self._drag_start
            p2 = end
            left = min(p1.x(), p2.x())
            right = max(p1.x(), p2.x())
            top = min(p1.y(), p2.y())
            bottom = max(p1.y(), p2.y())

            img1 = self._widget_to_image_xy(QPoint(left, top))
            img2 = self._widget_to_image_xy(QPoint(right, bottom))

            if img1 is None and img2 is None:
                return

            def clamp_widget_point_to_image(p: QPoint) -> tuple[int, int] | None:
                c = QPoint(self.width() // 2, self.height() // 2)
                q = QPoint(p)
                for _ in range(25):
                    xy = self._widget_to_image_xy(q)
                    if xy is not None:
                        return xy
                    q = QPoint(
                        int(q.x() + (c.x() - q.x()) * 0.2),
                        int(q.y() + (c.y() - q.y()) * 0.2)
                    )
                return None

            if img1 is None:
                img1 = clamp_widget_point_to_image(QPoint(left, top))
            if img2 is None:
                img2 = clamp_widget_point_to_image(QPoint(right, bottom))
            if img1 is None or img2 is None:
                return

            x1, y1 = img1
            x2, y2 = img2
            img_rect = QRect(min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1)

            avg, count = self._sample_rect_avg(img_rect)
            if count > 0:
                self.rectSampled.emit((avg, img_rect, count))

        self.update()

    def wheelEvent(self, e):
        if not self.has_image():
            return
        if e.modifiers() & Qt.ControlModifier:
            delta = e.angleDelta().y()
            if delta == 0:
                return
            factor = 1.12 if delta > 0 else 1 / 1.12
            self.set_scale(self._scale * factor)
            e.accept()
        else:
            super().wheelEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.has_image() or not self.pixmap():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self._dragging:
            r = QRect(self._drag_start, self._drag_end).normalized()
            pen = QPen(QColor(255, 0, 180))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(r)

        if self._last_mouse_pos.x() >= 0:
            p = self._last_mouse_pos
            pen = QPen(QColor(0, 180, 255))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(p.x() - 10, p.y(), p.x() + 10, p.y())
            painter.drawLine(p.x(), p.y() - 10, p.x(), p.y() + 10)

        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Color Picker (RGB / HEX / HEX8(RGBA) + HSV info)")

        self.canvas = ImageCanvas()
        self.canvas.setStyleSheet("background: #1f1f1f;")
        self.canvas.pixelSampled.connect(self.on_pixel_sampled)
        self.canvas.rectSampled.connect(self.on_rect_sampled)

        self.lbl_mode = QLabel("操作：クリック=ピクセル / ドラッグ=範囲平均 / Ctrl+ホイール=ズーム")
        self.lbl_mode.setWordWrap(True)

        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(5, 2000)
        self.zoom_spin.setValue(100)
        self.zoom_spin.setSuffix(" %")
        self.zoom_spin.valueChanged.connect(self.on_zoom_changed)

        btn_open = QPushButton("画像を開く")
        btn_open.clicked.connect(self.open_image)

        btn_copy_hex = QPushButton("HEXをコピー (#RRGGBB)")
        btn_copy_rgb = QPushButton("RGBをコピー")
        btn_copy_hex8 = QPushButton("RGBA HEXをコピー (#RRGGBBAA)")
        btn_copy_hsv_num = QPushButton("HSV数値をコピー")

        btn_copy_hex.clicked.connect(lambda: self.copy_text(self.hex_value.text()))
        btn_copy_rgb.clicked.connect(lambda: self.copy_text(self.rgb_value.text()))
        btn_copy_hex8.clicked.connect(lambda: self.copy_text(self.hex8_value.text()))
        btn_copy_hsv_num.clicked.connect(lambda: self.copy_text(self.hsv_num_value.text()))

        self.color_preview = QLabel()
        self.color_preview.setFixedHeight(48)
        self.color_preview.setStyleSheet("background: #000; border: 1px solid #444; border-radius: 6px;")

        self.rgb_value = QLabel("-")
        self.hex_value = QLabel("-")
        # ★ユーザー要望：ここに #RRGGBBAA を表示したいなら、この欄を見ればOK
        self.hex8_value = QLabel("-")
        # HSVの数値版（H,S,V）
        self.hsv_num_value = QLabel("-")
        self.extra_value = QLabel("-")
        self.extra_value.setWordWrap(True)

        grp = QGroupBox("取得した色")
        form = QFormLayout()
        form.addRow("RGB:", self.rgb_value)
        form.addRow("HEX (#RRGGBB):", self.hex_value)
        form.addRow("RGBA HEX (#RRGGBBAA):", self.hex8_value)
        form.addRow("HSV(数値):", self.hsv_num_value)
        form.addRow("補足:", self.extra_value)
        grp.setLayout(form)

        right = QVBoxLayout()
        right.addWidget(btn_open)
        right.addWidget(self.lbl_mode)
        right.addWidget(QLabel("ズーム："))
        right.addWidget(self.zoom_spin)
        right.addWidget(QLabel("プレビュー："))
        right.addWidget(self.color_preview)
        right.addWidget(grp)
        right.addWidget(btn_copy_hex)
        right.addWidget(btn_copy_rgb)
        right.addWidget(btn_copy_hex8)
        right.addWidget(btn_copy_hsv_num)
        right.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self.canvas, 1)
        rightw = QWidget()
        rightw.setLayout(right)
        rightw.setFixedWidth(360)
        root.addWidget(rightw, 0)

        c = QWidget()
        c.setLayout(root)
        self.setCentralWidget(c)

        open_act = QAction("画像を開く", self)
        open_act.triggered.connect(self.open_image)
        self.menuBar().addMenu("File").addAction(open_act)

        self._last_sample: ColorSample | None = None

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "画像を選択",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All Files (*.*)"
        )
        if not path:
            return
        try:
            self.canvas.load_image(path)
            self.zoom_spin.blockSignals(True)
            self.zoom_spin.setValue(100)
            self.zoom_spin.blockSignals(False)
            self.statusBar().showMessage(f"Loaded: {path}", 5000)
        except Exception as ex:
            QMessageBox.critical(self, "読み込みエラー", str(ex))

    def on_zoom_changed(self, percent: int):
        self.canvas.set_scale(percent / 100.0)

    def _apply_sample_to_ui(self, sample: ColorSample, extra: str = ""):
        self._last_sample = sample
        self.rgb_value.setText(f"{sample.r}, {sample.g}, {sample.b}")
        self.hex_value.setText(sample.hex)
        self.hex8_value.setText(sample.hex8)

        h, s, v = sample.hsv
        self.hsv_num_value.setText(f"H={h:.1f}°, S={s:.1f}%, V={v:.1f}%")

        self.extra_value.setText(extra if extra else "-")

        # preview uses RGB (alpha ignored by QLabel background)
        self.color_preview.setStyleSheet(
            f"background: {sample.hex}; border: 1px solid #444; border-radius: 6px;"
        )

    def on_pixel_sampled(self, sample: ColorSample):
        self._apply_sample_to_ui(sample, extra="ピクセル（クリック位置）")

    def on_rect_sampled(self, payload):
        sample, img_rect, count = payload
        extra = f"範囲平均：rect=({img_rect.left()},{img_rect.top()})-({img_rect.right()},{img_rect.bottom()}) / {count} px"
        self._apply_sample_to_ui(sample, extra=extra)

    def copy_text(self, text: str):
        if not text or text == "-":
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("コピーしました", 1500)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1260, 780)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()