"""
Generate every ArUco marker the project needs, as standalone PNGs and as one
printable PDF.

There are six markers, all from DICT_4X4_50 (the IDs come straight from
poa.config so they can never drift from what the tracker expects):

    IDs 0-3  workspace corners   (TL, TR, BR, BL)   -- default 40 mm
    ID  4    drone               (mount on top)     -- default 30 mm
    ID  5    landing pad         (lay on the floor) -- default 30 mm

The drone marker is printed with a FORWARD arrow above it. Stick it on the
drone with that arrow pointing at the nose; the tracker reads the marker's top
edge as the drone's heading, so getting this right is what makes the roll/pitch
commands come out in the correct direction.

Each PNG includes a white quiet zone so it can be detected straight off the
print. Each PDF page is drawn at the exact physical size with a 100 mm scale
bar, so you can check with a ruler that your printer didn't shrink-to-fit.

Usage:
    python generate_markers.py                       # PNGs in markers/, PDF too
    python generate_markers.py --marker-size 30 --corner-size 40
    python generate_markers.py --out custom.pdf --png-dir custom_dir
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from poa.config import ARUCO_DICT_NAME, CORNER_IDS, CORNER_NAMES, DRONE_ID, PAD_ID

DICT_ID = getattr(cv2.aruco, ARUCO_DICT_NAME)

# Render markers at 20 px/mm = ~508 DPI, comfortably above 300 DPI at any size.
PX_PER_MM = 20


@dataclass
class MarkerSpec:
    id: int
    title: str        # page header / png purpose
    placement: str    # one-line note on where it goes
    png_name: str
    size_mm: float
    forward_arrow: bool = False


def build_specs(corner_size_mm: float, marker_size_mm: float) -> list[MarkerSpec]:
    """
    Builds the list of MarkerSpec objects for the four workspace corners, the drone marker, and the landing-pad marker.
    
    Parameters:
        corner_size_mm (float): Side length in millimeters to assign to each workspace corner marker.
        marker_size_mm (float): Side length in millimeters to assign to the drone and landing-pad markers.
    
    Returns:
        list[MarkerSpec]: Six MarkerSpec instances in this order: the workspace corners (one per entry in CORNER_IDS/CORNER_NAMES, preserving their order), then the drone marker, then the landing-pad marker.
    """
    specs = [
        MarkerSpec(
            id=cid,
            title=f"Workspace corner {name}",
            placement=f"Lay flat at the {name} corner. Corner centres span the workspace.",
            png_name=f"corner_{cid}_{name}.png",
            size_mm=corner_size_mm,
        )
        for cid, name in zip(CORNER_IDS, CORNER_NAMES)
    ]
    specs.append(MarkerSpec(
        id=DRONE_ID,
        title="Drone marker",
        placement="Stick on top of the drone, arrow toward the nose.",
        png_name="drone_marker.png",
        size_mm=marker_size_mm,
        forward_arrow=True,
    ))
    specs.append(MarkerSpec(
        id=PAD_ID,
        title="Landing pad marker",
        placement="Lay flat on the floor where the drone should land.",
        png_name="landing_marker.png",
        size_mm=marker_size_mm,
    ))
    return specs


# --- bitmaps -------------------------------------------------------------

def marker_bitmap(marker_id: int, side_px: int) -> np.ndarray:
    """
    Generate an ArUco marker image without a quiet zone.
    
    Parameters:
        marker_id (int): ArUco marker identifier from the selected predefined dictionary.
        side_px (int): Length of the marker image side in pixels.
    
    Returns:
        marker (np.ndarray): Grayscale image array of shape (side_px, side_px) containing the marker bitmap.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)


def marker_png(spec: MarkerSpec) -> np.ndarray:
    """
    Create a printable BGR image containing the ArUco marker with a white quiet zone; optionally include a FORWARD arrow and caption for drone markers.
    
    Returns:
        image (np.ndarray): BGR image array (dtype=uint8, 0–255) sized to include the marker, surrounding quiet zone, and an optional arrow/caption area.
    """
    side_px = int(round(spec.size_mm * PX_PER_MM))
    quiet = side_px // 4
    arrow_zone = int(side_px * 0.6) if spec.forward_arrow else 0

    top = quiet + arrow_zone
    h = top + side_px + quiet
    w = side_px + 2 * quiet
    canvas_img = np.full((h, w, 3), 255, dtype=np.uint8)

    x0, y0 = quiet, top
    marker = cv2.cvtColor(marker_bitmap(spec.id, side_px), cv2.COLOR_GRAY2BGR)
    canvas_img[y0:y0 + side_px, x0:x0 + side_px] = marker

    if spec.forward_arrow:
        cx = w // 2
        tip_y = quiet // 2
        base_y = arrow_zone  # leaves a clean quiet-zone gap above the marker
        cv2.arrowedLine(canvas_img, (cx, base_y), (cx, tip_y),
                        (0, 0, 0), max(2, side_px // 60), cv2.LINE_AA, tipLength=0.35)
        cv2.putText(canvas_img, "FORWARD (nose)", (quiet, base_y + quiet // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, side_px / 700.0, (0, 0, 0),
                    max(1, side_px // 250), cv2.LINE_AA)
    return canvas_img


def save_pngs(specs: list[MarkerSpec], png_dir: Path) -> None:
    """
    Write PNG files for each MarkerSpec into the given directory, creating the directory if it does not exist.
    
    Parameters:
        specs (list[MarkerSpec]): Marker specifications to render and save as PNG files.
        png_dir (Path): Destination directory for output PNG files; it will be created if missing.
    
    Raises:
        RuntimeError: If writing any PNG file fails.
    """
    png_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = png_dir / spec.png_name
        if not cv2.imwrite(str(path), marker_png(spec)):
            raise RuntimeError(f"Failed to write {path}")
        print(f"  PNG  ID {spec.id:>2}  {int(spec.size_mm)} mm  -> {path}")


# --- printable PDF -------------------------------------------------------

def _png_bytes(spec: MarkerSpec, side_px: int) -> bytes:
    """
    Encode the raw ArUco marker bitmap for a spec into PNG bytes.
    
    Parameters:
        spec (MarkerSpec): Marker specification whose `id` selects the ArUco marker.
        side_px (int): Side length of the generated marker bitmap in pixels.
    
    Returns:
        bytes: PNG-encoded bytes of the bare marker bitmap (no quiet zone or annotations).
    
    Raises:
        RuntimeError: If PNG encoding fails.
    """
    ok, buf = cv2.imencode(".png", marker_bitmap(spec.id, side_px))
    if not ok:
        raise RuntimeError(f"Failed to encode marker {spec.id}")
    return buf.tobytes()


def draw_scale_bar(c: canvas.Canvas, cx_mm: float, y_mm: float, length_mm: float = 100.0) -> None:
    """
    Draws a horizontal scale bar in millimeters centered at the given x position on the PDF canvas.
    
    Parameters:
        c (canvas.Canvas): ReportLab canvas to draw onto.
        cx_mm (float): Center x-coordinate of the scale bar in millimeters.
        y_mm (float): Vertical position of the scale bar baseline in millimeters.
        length_mm (float): Total length of the scale bar in millimeters (defaults to 100.0).
    
    Behavior:
        - Renders a horizontal line of length `length_mm` centered at `cx_mm`.
        - Draws ticks every 10 mm; ticks at every 50 mm are taller.
        - Places "0" at the left end and "`{length_mm} mm`" at the right end.
        - Adds a centered caption indicating the bar should measure exactly `length_mm` mm on paper.
    """
    x0, x1 = cx_mm - length_mm / 2, cx_mm + length_mm / 2
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.6)
    c.line(x0 * mm, y_mm * mm, x1 * mm, y_mm * mm)
    for i in range(int(length_mm // 10) + 1):
        tx = x0 + i * 10
        tick = 3.5 if i % 5 == 0 else 2.0
        c.line(tx * mm, y_mm * mm, tx * mm, (y_mm + tick) * mm)
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x0 * mm, (y_mm - 4) * mm, "0")
    c.drawRightString(x1 * mm, (y_mm - 4) * mm, f"{int(length_mm)} mm")
    c.drawCentredString(cx_mm * mm, (y_mm - 9) * mm,
                        "Scale check: this bar should measure exactly 100 mm on paper.")


def _draw_forward_arrow(c: canvas.Canvas, cx_mm: float, base_y_mm: float, height_mm: float = 18.0) -> None:
    """
    Draws an upward-pointing arrow with a centered "FORWARD" caption above a marker on the PDF canvas.
    
    Parameters:
        c (canvas.Canvas): ReportLab canvas to draw onto.
        cx_mm (float): Horizontal center position of the arrow, in millimeters from the page origin.
        base_y_mm (float): Vertical position (base) of the arrow shaft, in millimeters from the page origin.
        height_mm (float): Vertical height of the arrow from base to tip, in millimeters (default 18.0).
    """
    tip = base_y_mm + height_mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    c.setLineWidth(1.5)
    c.line(cx_mm * mm, base_y_mm * mm, cx_mm * mm, (tip - 4) * mm)
    head = c.beginPath()
    head.moveTo(cx_mm * mm, tip * mm)
    head.lineTo((cx_mm - 3) * mm, (tip - 5) * mm)
    head.lineTo((cx_mm + 3) * mm, (tip - 5) * mm)
    head.close()
    c.drawPath(head, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(cx_mm * mm, (tip + 3) * mm, "FORWARD")


def draw_page(c: canvas.Canvas, spec: MarkerSpec) -> None:
    """
    Render a single A4 PDF page for the given marker specification and advance the canvas to a new page.
    
    Draws a centered title and subtitle, places the marker image at the specified physical size with a grey cut guide, optionally draws a "FORWARD" arrow above the marker, adds identifying text (ID, dictionary name and physical dimensions, placement note), draws a 100 mm scale bar, and calls showPage().
    
    Parameters:
        c (reportlab.pdfgen.canvas.Canvas): Canvas to draw onto.
        spec (MarkerSpec): Marker specification containing id, title, placement, size_mm, png_name, and forward_arrow flag.
    """
    page_w = A4[0] / mm  # 210
    page_h = A4[1] / mm  # 297

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_w / 2 * mm, (page_h - 18) * mm, spec.title)
    c.setFont("Helvetica", 9)
    c.drawCentredString(page_w / 2 * mm, (page_h - 26) * mm,
                        "Print at 100% scale (no shrink-to-fit). Cut on the grey guide, mount flat.")

    marker_x = (page_w - spec.size_mm) / 2
    marker_y = (page_h - spec.size_mm) / 2 + 10

    side_px = int(round(spec.size_mm * PX_PER_MM))
    img = ImageReader(io.BytesIO(_png_bytes(spec, side_px)))
    c.drawImage(img, marker_x * mm, marker_y * mm,
                width=spec.size_mm * mm, height=spec.size_mm * mm)

    pad = 2.0
    c.setLineWidth(0.2)
    c.setStrokeColorRGB(0.65, 0.65, 0.65)
    c.rect((marker_x - pad) * mm, (marker_y - pad) * mm,
           (spec.size_mm + 2 * pad) * mm, (spec.size_mm + 2 * pad) * mm)

    if spec.forward_arrow:
        _draw_forward_arrow(c, page_w / 2, marker_y + spec.size_mm + pad + 4)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(page_w / 2 * mm, (marker_y - 14) * mm, f"ID {spec.id}")
    c.setFont("Helvetica", 10)
    c.drawCentredString(page_w / 2 * mm, (marker_y - 22) * mm,
                        f"{ARUCO_DICT_NAME}   |   {int(spec.size_mm)} mm x {int(spec.size_mm)} mm")
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(page_w / 2 * mm, (marker_y - 31) * mm, spec.placement)

    draw_scale_bar(c, cx_mm=page_w / 2, y_mm=30.0)
    c.showPage()


def save_pdf(specs: list[MarkerSpec], out_path: Path) -> None:
    """
    Create an A4 PDF file containing one marker page per spec and write it to out_path.
    
    Ensures the output directory exists, creates a ReportLab A4 canvas (title and author metadata),
    renders each MarkerSpec as a separate page via draw_page, saves the PDF to disk, and prints a completion line.
    
    Parameters:
        specs (list[MarkerSpec]): Sequence of marker specifications; each entry becomes one PDF page.
        out_path (Path): Filesystem path where the generated PDF will be written; parent directories
            are created if they do not exist.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    c.setTitle("Psychic Object Avoidance - ArUco markers")
    c.setAuthor("generate_markers.py")
    for spec in specs:
        draw_page(c, spec)
    c.save()
    print(f"  PDF  {len(specs)} pages -> {out_path}")


def main() -> None:
    """
    Parse CLI arguments, validate sizes, build marker specifications, and generate standalone PNGs and a multi-page A4 PDF.
    
    The function accepts command-line options for corner marker size (--corner-size), drone/pad marker size (--marker-size),
    output PDF path (--out), and PNG output directory (--png-dir). It validates size constraints required to fit pages on A4;
    on invalid values it exits with a SystemExit. On success it constructs marker specifications, writes individual PNG files,
    and produces the printable A4 PDF containing one page per marker.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corner-size", type=float, default=40.0,
                        help="Corner marker side length in mm (default: 40).")
    parser.add_argument("--marker-size", type=float, default=30.0,
                        help="Drone and pad marker side length in mm (default: 30).")
    parser.add_argument("--out", type=Path, default=Path("markers/aruco_markers.pdf"),
                        help="Output PDF path (default: markers/aruco_markers.pdf).")
    parser.add_argument("--png-dir", type=Path, default=Path("markers"),
                        help="Directory for the standalone PNGs (default: markers/).")
    args = parser.parse_args()

    if not 10.0 <= args.corner_size <= 180.0:
        raise SystemExit("--corner-size must be between 10 and 180 mm to fit A4 with margins.")
    if not 10.0 <= args.marker_size <= 150.0:
        raise SystemExit("--marker-size must be between 10 and 150 mm "
                         "(the drone page needs headroom above the marker for the FORWARD arrow).")

    specs = build_specs(args.corner_size, args.marker_size)
    print(f"Generating {len(specs)} markers ({ARUCO_DICT_NAME}):")
    save_pngs(specs, args.png_dir)
    save_pdf(specs, args.out)


if __name__ == "__main__":
    main()
