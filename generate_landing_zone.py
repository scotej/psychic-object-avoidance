"""
Generate the big landing-zone ArUco marker (the pad, ID 5) at a real-world
size that the small per-marker generator can't reach.

`generate_markers.py` lays every marker out one-per-A4-page and caps the size at
150 mm because that is all that fits. The landing zone wants to be much bigger
than that -- the default here is a 300 mm x 300 mm (30 cm) marker -- so it needs
its own layout. The pipeline still detects it as PAD_ID (ID 5) from the same
DICT_4X4_50 dictionary; only the printed size changes, and detection is purely
2-D (homography), so a bigger pad is just an easier-to-see ID 5.

Three deliverables come out of one run:

  1. landing_zone.png            -- high-res raster with the print DPI baked in,
                                    handy for a digital pad on a screen or for a
                                    print shop that wants an image.
  2. landing_zone.pdf            -- ONE page sized exactly to the marker (plus its
                                    quiet zone). Send this to a large-format /
                                    plotter print, or "poster/tile" it from Acrobat.
  3. landing_zone_tiled_a4.pdf   -- the same marker sliced across overlapping A4
                                    pages for a home printer, with crop marks, an
                                    overlap band to glue on, a numbered assembly
                                    cover page, and a 100 mm scale-check bar.

A quiet zone (white border) is included because the pad lies on a floor that is
usually not white, and ArUco wants ~one module of clear space around the code to
detect it reliably. The default quiet zone is one module (size / 6 = 50 mm for a
300 mm marker). It is baked into landing_zone.png and the exact-size
landing_zone.pdf only; the A4 tiling covers just the marker face, so the tile
count depends solely on --size, --margin-mm and --overlap-mm, not on --quiet-mm.

Usage:
    python generate_landing_zone.py                      # 300 mm pad, all three files
    python generate_landing_zone.py --size 250
    python generate_landing_zone.py --quiet-mm 25 --overlap-mm 15
    python generate_landing_zone.py --id 5 --out-dir markers
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from poa.config import ARUCO_DICT_NAME, PAD_ID

DICT_ID = getattr(cv2.aruco, ARUCO_DICT_NAME)

# Master raster resolution. The marker is only 6 modules across, so edges stay
# razor-sharp at any printed size; 10 px/mm = 254 DPI keeps the file light while
# staying comfortably above the 150 DPI a print shop wants. Kept an integer so a
# size divisible by 6 (e.g. 300) lands module boundaries on exact pixels.
PX_PER_MM = 10

PAGE_W_MM = A4[0] / mm  # 210
PAGE_H_MM = A4[1] / mm  # 297


@dataclass
class TileLayout:
    """How the (marker + quiet zone) square maps onto a grid of A4 pages."""
    total_mm: float          # marker face + 2x quiet zone, the printed square
    margin_mm: float         # white margin kept clear at every page edge
    overlap_mm: float        # content duplicated between neighbouring pages
    usable_w_mm: float       # printable width per page (page - 2x margin)
    usable_h_mm: float       # printable height per page
    ncols: int
    nrows: int

    @property
    def ntiles(self) -> int:
        return self.ncols * self.nrows

    def _starts(self, n: int, usable: float) -> list[float]:
        """Left/top edge (in master-square mm) of each tile along one axis.

        Tiles step by (usable - overlap) so EVERY adjacent pair overlaps by
        exactly overlap_mm -- the amount the grey band and assembly steps assume.
        The last tile is left partial (a mostly-white page) rather than
        right-aligned, which would otherwise overlap its neighbour by far more
        than overlap_mm and break the trim-and-tuck instructions."""
        step = usable - self.overlap_mm
        return [i * step for i in range(n)]

    def col_spans(self) -> list[tuple[float, float]]:
        xs = self._starts(self.ncols, self.usable_w_mm)
        return [(x, min(x + self.usable_w_mm, self.total_mm)) for x in xs]

    def row_spans(self) -> list[tuple[float, float]]:
        ys = self._starts(self.nrows, self.usable_h_mm)
        return [(y, min(y + self.usable_h_mm, self.total_mm)) for y in ys]


def plan_tiles(total_mm: float, margin_mm: float, overlap_mm: float) -> TileLayout:
    usable_w = PAGE_W_MM - 2 * margin_mm
    usable_h = PAGE_H_MM - 2 * margin_mm
    if usable_w <= overlap_mm or usable_h <= overlap_mm:
        raise SystemExit("--margin-mm / --overlap-mm leave no usable page area.")

    def count(total: float, usable: float) -> int:
        if total <= usable:
            return 1
        step = usable - overlap_mm
        return int(math.ceil((total - overlap_mm) / step))

    return TileLayout(
        total_mm=total_mm,
        margin_mm=margin_mm,
        overlap_mm=overlap_mm,
        usable_w_mm=usable_w,
        usable_h_mm=usable_h,
        ncols=count(total_mm, usable_w),
        nrows=count(total_mm, usable_h),
    )


# --- master raster -------------------------------------------------------

def build_master(marker_id: int, size_mm: float, quiet_mm: float) -> np.ndarray:
    """The full printed square as a grayscale array: the marker centred in a
    white quiet zone, at PX_PER_MM. Module boundaries are pixel-exact when
    size_mm is a multiple of 6."""
    side_px = int(round(size_mm * PX_PER_MM))
    # Snap to a multiple of 6 so cv2 renders modules without resampling fuzz.
    side_px -= side_px % 6
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_ID)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)

    quiet_px = int(round(quiet_mm * PX_PER_MM))
    total_px = side_px + 2 * quiet_px
    canvas_img = np.full((total_px, total_px), 255, dtype=np.uint8)
    canvas_img[quiet_px:quiet_px + side_px, quiet_px:quiet_px + side_px] = marker
    return canvas_img


def save_png(master: np.ndarray, total_mm: float, path: Path) -> None:
    """Save the master with a print DPI derived from its *realized* pixel count,
    so the PNG opens at exactly total_mm no matter how module-snapping nudged the
    pixel size. (A fixed DPI would let the snap shrink the printed size slightly.)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    h_px = master.shape[0]
    dpi = h_px / total_mm * 25.4  # px/in that makes h_px span exactly total_mm
    Image.fromarray(master).save(path, dpi=(dpi, dpi))
    print(f"  PNG  {h_px}x{h_px}px @ {dpi:.0f} DPI ({total_mm:.0f} mm)  -> {path}")


# --- shared drawing helpers ---------------------------------------------

def _crop_to_imagereader(master: np.ndarray, total_mm: float,
                         x0: float, x1: float, y0: float, y1: float) -> ImageReader:
    """Crop the master to a mm window (y measured from the TOP) and hand it to
    reportlab as a PNG."""
    h_px = master.shape[0]
    per_mm = h_px / total_mm
    px0, px1 = int(round(x0 * per_mm)), int(round(x1 * per_mm))
    py0, py1 = int(round(y0 * per_mm)), int(round(y1 * per_mm))
    crop = master[py0:py1, px0:px1]
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError("Failed to encode tile crop.")
    return ImageReader(io.BytesIO(buf.tobytes()))


def draw_scale_bar(c: canvas.Canvas, cx_mm: float, y_mm: float, length_mm: float = 100.0) -> None:
    """Horizontal 100 mm scale bar with 10 mm ticks, for verifying print scale."""
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
                        "Scale check: this bar must measure exactly 100 mm on paper.")


# --- single-page, exact-size PDF (large format) -------------------------

def save_exact_pdf(master: np.ndarray, total_mm: float, size_mm: float,
                   marker_id: int, path: Path) -> None:
    """One page sized to the printed square, marker centred in its quiet zone.
    The page IS the marker -- no labels on it -- so a plotter prints it clean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=(total_mm * mm, total_mm * mm))
    c.setTitle(f"Landing-zone ArUco marker ID {marker_id} ({int(size_mm)} mm)")
    # The master already includes the quiet zone, so it fills the whole page;
    # the marker face then lands exactly size_mm wide, inset by quiet_mm.
    img = _crop_to_imagereader(master, total_mm, 0, total_mm, 0, total_mm)
    c.drawImage(img, 0, 0, width=total_mm * mm, height=total_mm * mm)
    c.showPage()
    c.save()
    print(f"  PDF  exact {total_mm:.0f}x{total_mm:.0f} mm single page  -> {path}")


# --- tiled A4 PDF (home printer) ----------------------------------------

def _draw_cover(c: canvas.Canvas, master: np.ndarray, layout: TileLayout,
                size_mm: float, quiet_mm: float, marker_id: int) -> None:
    cols, rows = layout.col_spans(), layout.row_spans()

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (PAGE_H_MM - 18) * mm,
                        "Landing-zone marker - assembly sheet")
    c.setFont("Helvetica", 10)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (PAGE_H_MM - 26) * mm,
                        f"{ARUCO_DICT_NAME}   |   ID {marker_id} (PAD)   |   "
                        f"marker face {int(size_mm)} x {int(size_mm)} mm")
    c.drawCentredString(PAGE_W_MM / 2 * mm, (PAGE_H_MM - 32) * mm,
                        f"tiled {layout.ncols} x {layout.nrows} = {layout.ntiles} A4 pages   "
                        f"|   then leave a {int(quiet_mm)} mm light border (quiet zone) around it")

    # Thumbnail of the marker face with the tile grid numbered the way the
    # following pages come out of the printer. Sized so the dashed quiet-zone
    # guide drawn around it clears both the header and the instructions below.
    thumb_mm = 84.0
    tx = (PAGE_W_MM - thumb_mm) / 2
    ty = 160.0
    img = _crop_to_imagereader(master, layout.total_mm, 0, layout.total_mm, 0, layout.total_mm)
    c.drawImage(img, tx * mm, ty * mm, width=thumb_mm * mm, height=thumb_mm * mm)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.rect(tx * mm, ty * mm, thumb_mm * mm, thumb_mm * mm)

    # Dashed guide showing the quiet zone to keep clear once assembled.
    q = quiet_mm * thumb_mm / layout.total_mm
    c.setStrokeColorRGB(0.2, 0.45, 0.85)
    c.setLineWidth(0.6)
    c.setDash(3, 2)
    c.rect((tx - q) * mm, (ty - q) * mm, (thumb_mm + 2 * q) * mm, (thumb_mm + 2 * q) * mm)
    c.setDash()
    c.setFillColorRGB(0.2, 0.45, 0.85)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawCentredString(PAGE_W_MM / 2 * mm, (ty - q - 4) * mm, "quiet zone - keep light/clear")

    scale = thumb_mm / layout.total_mm
    c.setStrokeColorRGB(0.85, 0.1, 0.1)
    c.setLineWidth(0.7)
    n = 1
    for r, (y0, y1) in enumerate(rows):
        for col, (x0, x1) in enumerate(cols):
            gx = tx + x0 * scale
            gw = (x1 - x0) * scale
            # thumbnail y is flipped (PDF origin bottom-left, square measured from top)
            gy = ty + (layout.total_mm - y1) * scale
            gh = (y1 - y0) * scale
            c.rect(gx * mm, gy * mm, gw * mm, gh * mm)
            c.setFillColorRGB(0.85, 0.1, 0.1)
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString((gx + gw / 2) * mm, (gy + gh / 2 - 1.5) * mm, str(n))
            n += 1

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, 132 * mm, "Print")
    c.setFont("Helvetica", 9.5)
    for i, line in enumerate([
        "1.  Print every page at 100% / Actual size  (NOT 'fit' or 'shrink to fit').",
        "2.  Check the 100 mm scale bar on a page below with a ruler before cutting.",
    ]):
        c.drawString(24 * mm, (126 - i * 5) * mm, line)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, 110 * mm, "Assemble")
    c.setFont("Helvetica", 9.5)
    for i, line in enumerate([
        "3.  Pages are numbered left-to-right, top-to-bottom (see grid above).",
        "4.  Each page has a solid crop rectangle and a grey overlap band on its",
        "     right / bottom edges. Trim the overlap band off the LEFT page and lay",
        "     it under the grey band of the page to its right / below, matching the art.",
        "5.  Tape or glue into one square -- that square is the marker face.",
        f"6.  Lay it on the floor with ~{int(quiet_mm)} mm of light, clear space on every",
        "     side (the dashed quiet zone above). Detection needs that clear border.",
    ]):
        c.drawString(24 * mm, (104 - i * 5) * mm, line)

    draw_scale_bar(c, cx_mm=PAGE_W_MM / 2, y_mm=30.0)
    c.showPage()


def _draw_tile(c: canvas.Canvas, master: np.ndarray, layout: TileLayout,
               x0: float, x1: float, y0: float, y1: float,
               row: int, col: int, index: int, with_scalebar: bool) -> None:
    w_mm, h_mm = x1 - x0, y1 - y0
    px = layout.margin_mm                      # content left on page
    py = PAGE_H_MM - layout.margin_mm - h_mm   # content bottom on page (top-aligned)

    img = _crop_to_imagereader(master, layout.total_mm, x0, x1, y0, y1)
    c.drawImage(img, px * mm, py * mm, width=w_mm * mm, height=h_mm * mm)

    # Solid crop rectangle around the printed content of this tile.
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.4)
    c.rect(px * mm, py * mm, w_mm * mm, h_mm * mm)

    # Grey overlap bands on the edges that a neighbour will sit under.
    ov = layout.overlap_mm
    if col < layout.ncols - 1:  # band on the RIGHT edge
        rx = px + w_mm - ov
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setLineWidth(0.3)
        c.line(rx * mm, py * mm, rx * mm, (py + h_mm) * mm)
        c.setFillColorRGB(0.6, 0.6, 0.6)
        c.setFont("Helvetica", 6)
        c.drawCentredString((rx + ov / 2) * mm, (py + h_mm / 2) * mm, "overlap")
    if row < layout.nrows - 1:  # band on the BOTTOM edge
        by = py + ov
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setLineWidth(0.3)
        c.line(px * mm, by * mm, (px + w_mm) * mm, by * mm)

    # Corner registration ticks on the content rectangle.
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.4)
    t = 4.0
    for cx, cy in [(px, py), (px + w_mm, py), (px, py + h_mm), (px + w_mm, py + h_mm)]:
        c.line((cx - t) * mm, cy * mm, (cx + t) * mm, cy * mm)
        c.line(cx * mm, (cy - t) * mm, cx * mm, (cy + t) * mm)

    # Page label in the top margin.
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(layout.margin_mm * mm, (PAGE_H_MM - layout.margin_mm + 1.5) * mm,
                 f"Tile {index} of {layout.ntiles}   (row {row + 1}, col {col + 1})")

    if with_scalebar:
        draw_scale_bar(c, cx_mm=PAGE_W_MM / 2, y_mm=layout.margin_mm / 2 + 4)
    c.showPage()


def save_tiled_pdf(master: np.ndarray, layout: TileLayout, size_mm: float,
                   quiet_mm: float, marker_id: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"Landing-zone ArUco marker ID {marker_id} - tiled A4")
    c.setAuthor("generate_landing_zone.py")

    _draw_cover(c, master, layout, size_mm, quiet_mm, marker_id)

    cols, rows = layout.col_spans(), layout.row_spans()
    index = 1
    for r, (y0, y1) in enumerate(rows):
        for col, (x0, x1) in enumerate(cols):
            # Put the scale bar on the first tile if there is clear margin under its art.
            with_bar = index == 1 and (PAGE_H_MM - layout.margin_mm - (y1 - y0)) > 16
            _draw_tile(c, master, layout, x0, x1, y0, y1, r, col, index, with_bar)
            index += 1
    c.save()
    print(f"  PDF  tiled {layout.ncols}x{layout.nrows} = {layout.ntiles} A4 pages  -> {path}")


# --- cli -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=float, default=300.0,
                        help="Marker face side length in mm (default: 300 = 30 cm).")
    parser.add_argument("--quiet-mm", type=float, default=None,
                        help="White quiet-zone border in mm (default: one module = size/6).")
    parser.add_argument("--id", type=int, default=PAD_ID,
                        help=f"ArUco ID (default: PAD_ID = {PAD_ID}).")
    parser.add_argument("--margin-mm", type=float, default=8.0,
                        help="Clear margin at every A4 page edge (default: 8).")
    parser.add_argument("--overlap-mm", type=float, default=12.0,
                        help="Overlap glued between A4 tiles (default: 12).")
    parser.add_argument("--out-dir", type=Path, default=Path("markers"),
                        help="Output directory (default: markers/).")
    args = parser.parse_args()

    if not 50.0 <= args.size <= 2000.0:
        raise SystemExit("--size must be between 50 and 2000 mm.")
    quiet_mm = args.quiet_mm if args.quiet_mm is not None else args.size / 6.0
    if quiet_mm < 0:
        raise SystemExit("--quiet-mm cannot be negative.")
    if not 0 <= args.id < 50:
        raise SystemExit("--id must be in 0..49 for DICT_4X4_50.")

    total_mm = args.size + 2 * quiet_mm
    # The single-page PDF and PNG bake the quiet zone in; the A4 tiling covers
    # only the marker face (so no page is wasted on blank quiet-zone border) and
    # asks the user to leave that border clear once the marker is on the floor.
    layout = plan_tiles(args.size, args.margin_mm, args.overlap_mm)

    print(f"Landing-zone marker  {ARUCO_DICT_NAME}  ID {args.id}")
    print(f"  face {args.size:.0f} mm + quiet {quiet_mm:.0f} mm "
          f"= {total_mm:.0f} mm square  |  A4 tiling of the face -> "
          f"{layout.ncols}x{layout.nrows} pages")

    full = build_master(args.id, args.size, quiet_mm)   # face + quiet zone
    face = build_master(args.id, args.size, 0.0)         # marker face only
    out = args.out_dir
    save_png(full, total_mm, out / "landing_zone.png")
    save_exact_pdf(full, total_mm, args.size, args.id, out / "landing_zone.pdf")
    save_tiled_pdf(face, layout, args.size, quiet_mm, args.id, out / "landing_zone_tiled_a4.pdf")


if __name__ == "__main__":
    main()
