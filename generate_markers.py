"""
Generate the 4 corner ArUco markers for the workspace, as a printable PDF.

One marker per page on A4. Each marker is rendered at an exact physical size
(default 100 mm x 100 mm) and labelled with its ID and intended workspace
corner. Every page also includes a 100 mm scale bar so you can hold a ruler
against the printout and confirm the printer didn't shrink-to-fit.

Usage:
    python generate_markers.py                       # markers/aruco_markers.pdf, 100 mm
    python generate_markers.py --size 150            # 150 mm markers
    python generate_markers.py --out custom.pdf
"""

import argparse
import io
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# Marker ID -> (long corner name, short label). The IDs are placed at these
# corners of the workspace so the rectification homography always has a
# consistent orientation.
CORNERS: dict[int, tuple[str, str]] = {
    0: ("Top-Left", "TL"),
    1: ("Top-Right", "TR"),
    2: ("Bottom-Right", "BR"),
    3: ("Bottom-Left", "BL"),
}

DICT_ID = cv2.aruco.DICT_4X4_50
DICT_NAME = "DICT_4X4_50"

# Render the marker bitmap at ~12 px/mm (>= 300 DPI when printed at 100 mm).
PX_PER_MM = 12


def make_marker_png_bytes(marker_id: int, side_px: int) -> bytes:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"Failed to encode marker {marker_id} as PNG")
    return buf.tobytes()


def draw_scale_bar(c: canvas.Canvas, cx_mm: float, y_mm: float, length_mm: float = 100.0) -> None:
    """Horizontal scale bar centered at cx_mm, baseline at y_mm. Ticks every 10 mm."""
    x0_mm = cx_mm - length_mm / 2
    x1_mm = cx_mm + length_mm / 2

    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.6)
    c.line(x0_mm * mm, y_mm * mm, x1_mm * mm, y_mm * mm)

    for i in range(int(length_mm // 10) + 1):
        tx_mm = x0_mm + i * 10
        tick_h_mm = 3.5 if i % 5 == 0 else 2.0
        c.line(tx_mm * mm, y_mm * mm, tx_mm * mm, (y_mm + tick_h_mm) * mm)

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x0_mm * mm, (y_mm - 4) * mm, "0")
    c.drawRightString(x1_mm * mm, (y_mm - 4) * mm, f"{int(length_mm)} mm")
    c.drawCentredString(
        cx_mm * mm, (y_mm - 9) * mm,
        "Scale check: this bar should measure exactly 100 mm on the printed page.",
    )


def draw_page(c: canvas.Canvas, marker_id: int, size_mm: float) -> None:
    page_w_mm = A4[0] / mm  # 210
    page_h_mm = A4[1] / mm  # 297

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w_mm / 2 * mm, (page_h_mm - 18) * mm, "Workspace ArUco marker")
    c.setFont("Helvetica", 9)
    c.drawCentredString(
        page_w_mm / 2 * mm, (page_h_mm - 26) * mm,
        "Print at 100% scale (no shrink-to-fit). Cut on the grey guide, mount flat at the assigned corner.",
    )

    # Marker, centered horizontally and vertically in the page
    marker_x_mm = (page_w_mm - size_mm) / 2
    marker_y_mm = (page_h_mm - size_mm) / 2 + 10  # nudge up to leave room for label + scale bar

    side_px = int(round(size_mm * PX_PER_MM))
    img = ImageReader(io.BytesIO(make_marker_png_bytes(marker_id, side_px)))
    c.drawImage(
        img,
        marker_x_mm * mm, marker_y_mm * mm,
        width=size_mm * mm, height=size_mm * mm,
    )

    # Light grey cut guide just outside the marker
    pad_mm = 2.0
    c.setLineWidth(0.2)
    c.setStrokeColorRGB(0.65, 0.65, 0.65)
    c.rect(
        (marker_x_mm - pad_mm) * mm, (marker_y_mm - pad_mm) * mm,
        (size_mm + 2 * pad_mm) * mm, (size_mm + 2 * pad_mm) * mm,
    )

    # Label directly under the marker
    long_name, short_name = CORNERS[marker_id]
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(
        page_w_mm / 2 * mm, (marker_y_mm - 14) * mm,
        f"ID {marker_id}   {long_name}  ({short_name})",
    )
    c.setFont("Helvetica", 10)
    c.drawCentredString(
        page_w_mm / 2 * mm, (marker_y_mm - 22) * mm,
        f"{DICT_NAME}   |   {int(size_mm)} mm x {int(size_mm)} mm",
    )

    # Scale verification bar near the bottom of the page
    draw_scale_bar(c, cx_mm=page_w_mm / 2, y_mm=30.0)

    c.showPage()


def generate(out_path: Path, size_mm: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle("Psychic Object Avoidance - Workspace ArUco markers")
    c.setAuthor("generate_markers.py")
    for marker_id in sorted(CORNERS):
        draw_page(c, marker_id, size_mm)
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--size", type=float, default=100.0,
        help="Marker side length in millimetres (default: 100).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("markers/aruco_markers.pdf"),
        help="Output PDF path (default: markers/aruco_markers.pdf).",
    )
    args = parser.parse_args()

    if args.size <= 0 or args.size > 180:
        raise SystemExit("--size must be between 0 and 180 mm to fit on A4 with margins.")

    generate(args.out, args.size)
    print(f"Wrote {len(CORNERS)} markers ({int(args.size)} mm, {DICT_NAME}) -> {args.out}")


if __name__ == "__main__":
    main()
