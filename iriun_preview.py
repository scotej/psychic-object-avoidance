"""
Iriun iPhone camera preview.

Milestone 1: confirm we can pull frames from the iPhone (via Iriun Webcam)
into OpenCV reliably.

Usage:
    python iriun_preview.py              # open camera index 0
    python iriun_preview.py --camera 1   # open a specific index
    python iriun_preview.py --list       # probe indices 0..5 and report what's there
"""

import argparse
import time

import cv2


def list_cameras(max_index: int = 5) -> None:
    print(f"Probing camera indices 0..{max_index} (DirectShow backend):")
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  [{idx}] not available")
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  [{idx}] opened but returned no frame")
        else:
            h, w = frame.shape[:2]
            print(f"  [{idx}] OK -- {w}x{h}")
        cap.release()


def run_preview(camera_index: int) -> None:
    # DirectShow plays nicest with virtual cameras like Iriun on Windows.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera index {camera_index}. "
            "Run with --list to find the right one."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {camera_index} opened at {width}x{height}. Press q or ESC to quit.")

    frames_in_window = 0
    fps = 0.0
    window_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame grab failed; retrying...")
            continue

        frames_in_window += 1
        elapsed = time.time() - window_start
        if elapsed >= 1.0:
            fps = frames_in_window / elapsed
            frames_in_window = 0
            window_start = time.time()

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"{fps:5.1f} FPS  {w}x{h}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Iriun preview (q or ESC to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the Iriun iPhone camera in OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--list", action="store_true", help="List available camera indices and exit")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    run_preview(args.camera)


if __name__ == "__main__":
    main()
