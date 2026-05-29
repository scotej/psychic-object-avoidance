# Psychic Object Avoidance — CoDrone Vision-Guided Landing

Vision-guided autonomous landing for a CoDrone EDU using an overhead iPhone camera. Everything is found by ArUco marker: four markers bound the workspace, one marker rides on the drone (giving both its position and its heading), and one marker on the floor is the landing pad. The system commands the drone toward the pad in real time, and a single live overlay window shows the rectified workspace for debugging and tuning.

## System Overview

One overhead iPhone camera looks straight down at a small workspace — roughly 30 × 30 cm. Four ArUco markers (one per corner) define the world frame, so the workspace is rectified to a known coordinate system on every frame; that way camera bumps, drift, or remounts don't break the geometry, and pixel positions map cleanly to a fixed mm-scale world.

Every object is an ArUco marker from the same dictionary, so there is no colour calibration to drift:
- The four **corner** markers (IDs 0–3) define the workspace.
- The **drone** marker (ID 4) is stuck on top of the drone with its top edge pointing at the nose. Detecting it gives the drone's position *and* its heading from the marker's corner geometry.
- The **pad** marker (ID 5) lies on the floor and is the landing target.
- A **controller** converts `(drone_pose, pad_pos)` → CoDrone flight commands and closes the loop until the drone is over the pad and can descend.

## Hardware
- **Drone:** CoDrone EDU
- **Camera:** iPhone, mounted overhead, looking straight down
- **Workspace markers:** 4× printed ArUco markers (one per corner), centres ~30 cm apart
- **Drone marker:** a 30 mm ArUco marker (ID 4) on top of the drone, top edge aligned with the nose
- **Landing target:** a 30 mm ArUco marker (ID 5) on the floor

## Per-Frame Pipeline (target ~15–30 Hz)
1. Grab frame from iPhone camera.
2. Detect every ArUco marker in one pass on the raw frame.
3. From the 4 corner markers → compute homography → rectify the workspace to a known mm/px scale.
4. Push the drone marker through the homography → drone `(x, y)` + heading in workspace coordinates.
5. Push the pad marker through the homography → pad `(x, y)` in workspace coordinates.
6. Controller: `error = pad - drone`, rotated into the drone's body frame → roll / pitch / yaw / throttle command.
7. Send command to the CoDrone via `codrone-edu`.
8. Draw the single overlay (markers, heading arrow, error vector, FPS) and display.

## Recommended Open Source Stack

### Core CV / control
- **OpenCV** — `pip install opencv-contrib-python`. Provides `cv2.aruco` (marker detection + generation), `cv2.getPerspectiveTransform` / `cv2.warpPerspective` (rectification), `cv2.perspectiveTransform` (mapping marker corners into the workspace), and overlay drawing. The `contrib` build is required for `cv2.aruco`.
- **NumPy** — array math, coordinate transforms.
- **reportlab** — lays the printable marker PDF out at an exact physical size (see `generate_markers.py`).

### Drone control
- **codrone-edu** — `pip install codrone-edu`. Official Python SDK for the CoDrone EDU. Gives you `Drone()` with `pair()`, `takeoff()`, `set_pitch()`, `set_roll()`, `set_throttle()`, `set_yaw()`, `move()`, `land()`, plus telemetry.

### iPhone → Python frame source (pick one)
Goal: make `cv2.VideoCapture(...)` return iPhone frames so the rest of the pipeline doesn't care where the camera lives.

- **Iriun Webcam** *(recommended starting point)* — iOS app + free Windows companion. Once both are running on the same network, the iPhone shows up as a normal system webcam, so `cv2.VideoCapture(0)` (or `1`) Just Works. No extra parsing code, low latency, easiest path.
- **IP Camera Lite (iOS) → MJPEG over HTTP** — turns the iPhone into an MJPEG server on your LAN. Read directly with `cv2.VideoCapture("http://<phone-ip>:8080/video")`. Good if you don't want a Windows helper app.
- **Larix Broadcaster (iOS) → RTSP/RTMP** — push RTSP and read with `cv2.VideoCapture("rtsp://...")` (OpenCV must be built against FFmpeg, which the standard wheels are). More setup; better if you want to record/restream later.
- **NDI HX Camera (iOS) + NDI SDK** — broadcast-grade, very low latency, heavier integration. Overkill unless the simpler options hit a latency wall.

Start with Iriun; only escalate if latency or wireless requirements force it.

### Optional / later
- **Ultralytics YOLOv8** — drop in if marker detection turns out to be too brittle (motion blur, glare, marker occlusion) and you want a learned detector for the drone and pad.
- **Stable-Baselines3 + Gymnasium** — for an actually-learned controller instead of a hand-tuned PID. Wrap perception + drone as a Gym env; train in sim or a digital twin first.
- **Flask or FastAPI + MJPEG endpoint** — if you want the live overlay viewable from a phone or browser instead of a local `cv2.imshow` window.
- **PyQt / Dear PyGui** — if the live feed grows into a real control panel with tuning sliders, telemetry plots, start/stop buttons.

## Suggested Milestones
1. **Camera in.** iPhone → Iriun → `cv2.VideoCapture` → `cv2.imshow`. Confirm a stable frame stream (`iriun_preview.py`).
2. **Markers out.** Print the markers from `generate_markers.py`, lay the four corners ~30 cm apart, stick ID 4 on the drone (arrow to the nose) and drop ID 5 on the floor.
3. **Workspace rectification.** Detect the four corners, compute the homography, and continuously visualize the rectified workspace.
4. **Detect the drone + pad.** Push markers 4 and 5 through the homography → positions + heading drawn on the overlay (`test_synthetic.py` checks this without hardware).
5. **CoDrone hello-world.** Connect, takeoff, hover, land via `codrone-edu` — no vision yet.
6. **Closed-loop landing.** Wire perception → proportional controller → drone (`land.py`). Hard safety `land()` on detection loss.
7. **Polish + tune.** Overlay shows the heading arrow, error vector, command vector, and FPS; tune the controller gains in `poa/config.py`.
8. *(stretch)* Swap ArUco for a learned detector (YOLO), or the hand-tuned controller for RL.

## Safety Notes
- Hard-code an emergency `land()` on: detection loss for more than N consecutive frames, drone leaving workspace bounds, keyboard interrupt.
- Cap max horizontal command magnitude until the controller is tuned.
- Test indoors with clear airspace above the iPhone and around the workspace.
