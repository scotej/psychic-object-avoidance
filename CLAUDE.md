# Psychic Object Avoidance — CoDrone Vision-Guided Landing

Vision-guided autonomous landing for a CoDrone EDU using an overhead iPhone camera. The system finds the drone (by its onboard LED color), finds a blue landing pad, and commands the drone toward the pad in real time. A live overlay window shows the rectified workspace for debugging and tuning.

## System Overview

One overhead iPhone camera looks straight down at the workspace. Four ArUco markers (one per corner) define the world frame so the workspace is rectified to a known coordinate system on every frame — that way camera bumps, drift, or remounts don't break the geometry, and pixel positions map cleanly to a fixed mm-scale world.

Within the rectified frame:
- The **drone** is located by detecting its **LED color** (configurable HSV range).
- The **landing pad** is located by detecting **blue**.
- A **controller** converts `(drone_pos, pad_pos)` → CoDrone flight commands and closes the loop until the drone is over the pad and can descend.

## Hardware
- **Drone:** CoDrone EDU
- **Camera:** iPhone, mounted overhead, looking straight down
- **Workspace markers:** 4× printed ArUco markers (one per corner)
- **Drone marker:** the drone's onboard LED (color chosen to be distinctive against the floor)
- **Landing target:** a blue physical landing pad on the floor

## Per-Frame Pipeline (target ~15–30 Hz)
1. Grab frame from iPhone camera.
2. Detect the 4 ArUco markers → compute homography → rectify the workspace to a known mm/px scale.
3. Detect the drone LED color blob → drone `(x, y)` in workspace coordinates.
4. Detect the blue pad blob → pad `(x, y)` in workspace coordinates.
5. Controller: `error = pad - drone` → roll / pitch / yaw / throttle command.
6. Send command to the CoDrone via `codrone-edu`.
7. Draw the overlay (markers, drone, pad, error vector, FPS) and display.

## Recommended Open Source Stack

### Core CV / control
- **OpenCV** — `pip install opencv-contrib-python`. Provides `cv2.aruco` (marker detection), `cv2.findHomography` / `cv2.warpPerspective` (rectification), HSV color segmentation, contour finding, and overlay drawing. The `contrib` build is required for `cv2.aruco`.
- **NumPy** — array math, coordinate transforms.

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
- **Ultralytics YOLOv8** — drop in if classical HSV blob detection turns out to be too brittle (reflections, ambient light changes) and you want a learned detector for the drone and pad.
- **Stable-Baselines3 + Gymnasium** — for an actually-learned controller instead of a hand-tuned PID. Wrap perception + drone as a Gym env; train in sim or a digital twin first.
- **Flask or FastAPI + MJPEG endpoint** — if you want the live overlay viewable from a phone or browser instead of a local `cv2.imshow` window.
- **PyQt / Dear PyGui** — if the live feed grows into a real control panel with tuning sliders, telemetry plots, start/stop buttons.

## Suggested Milestones
1. **Camera in.** iPhone → Iriun → `cv2.VideoCapture` → `cv2.imshow`. Confirm a stable frame stream.
2. **Workspace rectification.** Print 4 ArUco markers, lay them at the corners, compute and continuously visualize the rectified workspace.
3. **Detect the drone.** LED color → blob → `(x, y)` drawn on the overlay. Build a small HSV calibration script with trackbars.
4. **Detect the pad.** Same approach for blue, drawn on the overlay.
5. **CoDrone hello-world.** Connect, takeoff, hover, land via `codrone-edu` — no vision yet.
6. **Closed-loop landing.** Wire perception → simple proportional controller → drone. Hard safety `land()` on detection loss.
7. **Polish + tune.** Overlay shows error vector, command vector, FPS; HSV thresholds tunable from sliders.
8. *(stretch)* Swap classical CV for YOLO, or hand-tuned PID for RL.

## Safety Notes
- Hard-code an emergency `land()` on: detection loss for more than N consecutive frames, drone leaving workspace bounds, keyboard interrupt.
- Cap max horizontal command magnitude until the controller is tuned.
- Test indoors with clear airspace above the iPhone and around the workspace.
