"""Local visual QA helper. Captures the app's own window without UI automation."""

from pathlib import Path
import sys

from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modern_app import VisionLab


OUTPUT = ROOT / "qa"
OUTPUT.mkdir(exist_ok=True)


app = VisionLab()
app.geometry("1680x960+80+40")
app.attributes("-topmost", True)
app.lift()
app.focus_force()


states = [
    ("ui-b1.png", "CÀI ĐẶT MẪU", "B1 · Tách 2 mẫu vải", None),
    ("ui-b1-bottom.png", "CÀI ĐẶT MẪU", "B1 · Tách 2 mẫu vải", None),
    ("ui-b2.png", "CÀI ĐẶT MẪU", "B2 · Vị trí tương đối", None),
    ("ui-b3.png", "CÀI ĐẶT MẪU", "B3 · Điểm TCP & ROI", None),
    ("ui-operation.png", "VẬN HÀNH", None, None),
    ("ui-operation-robot.png", "VẬN HÀNH", None, None),
    ("ui-camera.png", "CAMERA BASLER", None, None),
    ("ui-learn-threshold.png", "HỌC VISION", None, 2),
    ("ui-learn-orb.png", "HỌC VISION", None, 8),
    ("ui-learn-pick-place.png", "HỌC VISION", None, 11),
]


def capture(index=0):
    if index >= len(states):
        app._close()
        return
    filename, main_tab, design_tab, lesson_index = states[index]
    app.tabs.set(main_tab)
    if design_tab:
        app.design_tabs.set(design_tab)
    if lesson_index is not None:
        app.lessons.selection_clear(0, "end")
        app.lessons.selection_set(lesson_index)
        app.lessons.see(lesson_index)
        app._lesson(None)
        if app._lesson_job:
            app.after_cancel(app._lesson_job)
            app._lesson_job = None
        app._render_current_lesson()
    if filename == "ui-b1.png":
        app.b1_scroll._parent_canvas.yview_moveto(0)
    elif filename == "ui-b1-bottom.png":
        app.b1_scroll._parent_canvas.yview_moveto(1)
    if main_tab == "VẬN HÀNH":
        app._detect()
        mode = "Robot · DeltaX" if filename == "ui-operation-robot.png" else "Ảnh tĩnh"
        app.source_selector.set(mode)
        app._source_mode_changed(mode)
    app.update_idletasks()
    app.after(350, lambda: save(index, filename))


def save(index, filename):
    app.lift()
    app.focus_force()
    app.update_idletasks()
    left, top = app.winfo_rootx(), app.winfo_rooty()
    right, bottom = left + app.winfo_width(), top + app.winfo_height()
    ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).save(OUTPUT / filename)
    print(OUTPUT / filename)
    capture(index + 1)


app.after(700, capture)
app.mainloop()
