<div align="center">

# 🏋️ AI Squat Form Analyzer

### Real-Time Squat Detection, Rep Counting & Form Feedback Using Computer Vision

**Made by Vidit Sharma**

A real-time computer vision system that uses **MediaPipe Pose Landmarker** and **OpenCV** to analyze squat movements through a webcam, calculate joint angles, count repetitions, and provide live feedback on squat depth and back posture.

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?style=for-the-badge\&logo=google\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Pose%20Estimation-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📖 Overview

**AI Squat Form Analyzer** is a real-time computer vision project designed to analyze squat movements using a standard webcam.

The system detects the user's body pose using **MediaPipe Pose Landmarker**, extracts important body landmarks, calculates **knee and hip joint angles**, identifies squat phases, counts completed repetitions, and provides real-time feedback about squat depth and posture.

Instead of relying on simple angle thresholds alone, the project uses a **state-machine-based squat detection system** with posture validation to make repetition counting more reliable.

### 🎯 What It Does

* Detects the human body in real time
* Tracks major lower-body and upper-body landmarks
* Calculates knee and hip angles
* Detects the transition between standing and squatting
* Counts completed squat repetitions
* Evaluates squat depth
* Checks back posture
* Detects potentially incorrect sitting-like postures
* Provides live visual feedback
* Displays real-time FPS
* Smooths pose landmarks using an Exponential Moving Average (EMA)

---

## ✨ Features

| Feature                         | Description                                                                   |
| ------------------------------- | ----------------------------------------------------------------------------- |
| 🧍 **Real-Time Pose Detection** | Detects a person's body landmarks using MediaPipe Pose Landmarker             |
| 📐 **Joint Angle Analysis**     | Calculates knee and hip angles using 3-point geometry                         |
| 🔢 **Automatic Rep Counting**   | Counts completed squat repetitions using a state machine                      |
| 🎯 **Squat Depth Detection**    | Determines whether the squat reaches the required depth                       |
| 🧍 **Back Posture Analysis**    | Detects excessive forward bending using hip angle                             |
| 🚫 **Sitting Detection**        | Rejects chair-like sitting movements based on ankle/hip alignment             |
| 📊 **Live HUD**                 | Displays reps, angles, feedback and FPS directly on the camera feed           |
| 🟢 **Form Feedback**            | Provides feedback such as `Good Squat!`, `Go Lower`, and `Keep Back Straight` |
| 🔄 **EMA Smoothing**            | Reduces jitter in detected landmark positions                                 |
| 🎥 **Webcam-Based**             | Works using a standard built-in or external webcam                            |
| ⚡ **VIDEO Mode**                | Uses MediaPipe's temporal tracking mode for real-time processing              |

---

## 🧠 How It Works

The application follows a modular computer vision pipeline:

```text
                    ┌─────────────────────┐
                    │       Webcam        │
                    │      OpenCV         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Pose Detection    │
                    │    MediaPipe       │
                    │   Pose Landmarker   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Landmark Extraction │
                    │ Shoulder / Hip /    │
                    │ Knee / Ankle        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   EMA Smoothing     │
                    │ Reduce Pose Jitter  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Angle Calculator   │
                    │ Knee + Hip Angles   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Squat Analyzer    │
                    │ State Machine +     │
                    │ Form Validation     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Live Feedback HUD  │
                    │ Reps / Angles / FPS │
                    └─────────────────────┘
```

### 🔄 Frame Processing Pipeline

Each webcam frame goes through the following process:

1. **Capture** — OpenCV captures a frame from the webcam.
2. **Mirror** — The frame is horizontally flipped for natural interaction.
3. **Pose Detection** — MediaPipe detects the person's body landmarks.
4. **Landmark Extraction** — Shoulder, hip, knee and ankle coordinates are extracted.
5. **Smoothing** — EMA filtering reduces landmark jitter.
6. **Angle Calculation** — Knee and hip angles are calculated.
7. **Posture Validation** — The system checks squat depth, back position and ankle/hip alignment.
8. **State Detection** — The squat state transitions between `IDLE`, `STANDING`, and `SQUATTING`.
9. **Rep Counting** — A complete squat cycle is counted as one repetition.
10. **Feedback Rendering** — Angles, skeleton, feedback, reps and FPS are displayed.

---

# 🤖 AI Model — MediaPipe Pose Landmarker

The project uses Google's **MediaPipe Pose Landmarker** through the MediaPipe Tasks API.

The detector identifies human body landmarks from the webcam feed and provides normalized coordinates and visibility scores.

### Model Configuration

| Parameter                       | Value                       |
| ------------------------------- | --------------------------- |
| **API**                         | MediaPipe Tasks API         |
| **Model**                       | `pose_landmarker_full.task` |
| **Fallback Model**              | `pose_landmarker_lite.task` |
| **Running Mode**                | `VIDEO`                     |
| **Maximum Poses**               | 1                           |
| **Detection Confidence**        | 0.7                         |
| **Tracking Confidence**         | 0.7                         |
| **Minimum Landmark Visibility** | 0.65                        |
| **Smoothing**                   | Exponential Moving Average  |
| **Smoothing Factor**            | 0.4                         |

The application automatically looks for the **Full model first** and falls back to the **Lite model** if the full model is unavailable.

---

## 🦴 Body Landmarks Used

The project focuses on the following landmarks:

| Landmark       | Purpose                           |
| -------------- | --------------------------------- |
| Left Shoulder  | Hip angle calculation             |
| Right Shoulder | Hip angle calculation             |
| Left Hip       | Hip and knee analysis             |
| Right Hip      | Hip and knee analysis             |
| Left Knee      | Knee angle calculation            |
| Right Knee     | Knee angle calculation            |
| Left Ankle     | Knee angle and posture validation |
| Right Ankle    | Knee angle and posture validation |

The left and right coordinates are averaged to produce more stable measurements.

```text
       Left Shoulder ───── Right Shoulder
              │                   │
              │                   │
          Left Hip ─────────── Right Hip
              │                   │
              │                   │
         Left Knee ────────── Right Knee
              │                   │
              │                   │
        Left Ankle ───────── Right Ankle
```

---

# 📐 Angle Calculation

The project calculates joint angles using three-point geometry.

Given three points:

```text
A ───── B ───── C
        ↑
      Vertex
```

The angle is calculated at point **B** using vectors:

```text
BA = A - B
BC = C - B
```

The dot product and vector magnitudes are then used to calculate the angle in degrees.

### Knee Angle

The knee angle is calculated using:

```text
Hip → Knee → Ankle
```

```python
knee_angle = get_knee_angle(hip, knee, ankle)
```

A fully extended leg typically produces a large angle, while bending the knee reduces the angle.

### Hip Angle

The hip angle is calculated using:

```text
Shoulder → Hip → Knee
```

```python
hip_angle = get_hip_angle(shoulder, hip, knee)
```

This angle is used primarily to evaluate excessive forward bending of the torso.

---

# 🧠 Squat Detection Logic

Rather than counting repetitions whenever the knee angle crosses a single threshold, the project uses a **three-state machine**.

```text
             Standing Confirmed
        ┌─────────────────────────┐
        │                         ▼
     ┌───────┐              ┌──────────┐
     │ IDLE  │─────────────▶│ STANDING │
     └───────┘              └─────┬────┘
                                  │
                           Knee < 120°
                                  │
                                  ▼
                           ┌────────────┐
                           │ SQUATTING  │
                           └─────┬──────┘
                                 │
                          Knee > 155°
                                 │
                                 ▼
                           ┌──────────┐
                           │ STANDING │
                           └──────────┘
                                +
                              1 REP
```

### States

#### `IDLE`

The system waits until the user demonstrates a confirmed upright standing posture.

The user must remain above the standing knee-angle threshold for **10 consecutive frames** before the analyzer enters the `STANDING` state.

#### `STANDING`

The system is ready for the user to begin a squat.

When the knee angle falls below the squat threshold, the system transitions into `SQUATTING`.

#### `SQUATTING`

The system tracks:

* Minimum knee angle
* Hip angle
* Back posture
* Ankle/hip alignment

When the user returns to the standing position, the repetition is completed and the form is evaluated.

---

# 🎯 Squat Form Rules

The analyzer uses several configurable thresholds.

| Parameter                 | Value | Purpose                             |
| ------------------------- | ----: | ----------------------------------- |
| `KNEE_ANGLE_SQUAT`        |  120° | Starts squat detection              |
| `KNEE_ANGLE_STAND`        |  155° | Detects return to standing          |
| `KNEE_ANGLE_DEEP`         |  100° | Determines sufficient squat depth   |
| `HIP_ANGLE_MIN`           |   55° | Excessive forward bending           |
| `HIP_ANGLE_GOOD_MIN`      |   65° | Acceptable back posture             |
| `STANDING_CONFIRM_FRAMES` |    10 | Required standing frames            |
| `MAX_ANKLE_HIP_X_RATIO`   |   20% | Sitting-posture rejection threshold |

---

# 💬 Real-Time Feedback

The system provides live feedback based on the detected movement.

### During Squat

| Feedback                   | Meaning                         |
| -------------------------- | ------------------------------- |
| **Ready — begin squat**    | Correct starting position       |
| **Go Lower**               | Squat depth is insufficient     |
| **Good Depth — Hold!**     | Sufficient squat depth achieved |
| **Straighten Back**        | Back posture needs improvement  |
| **Keep Back Straight!**    | Excessive forward bending       |
| **Stand Up — not a squat** | Sitting-like posture detected   |

### After Completing a Rep

| Feedback                  | Condition                                  |
| ------------------------- | ------------------------------------------ |
| 🟢 **Good Squat!**        | Deep enough + acceptable back posture      |
| 🟡 **Go Lower**           | Squat completed but depth was insufficient |
| 🔴 **Keep Back Straight** | Squat completed with poor back posture     |

The repetition is still counted after a completed squat cycle, while the feedback indicates what should be improved.

---

# 🚫 Sitting Posture Rejection

A major part of the detection logic is designed to prevent a user from artificially increasing the repetition count by sitting down.

The system compares the horizontal distance between the ankle and hip positions.

If:

* The ankle is significantly displaced from the hip horizontally
* And the ankle position indicates a forward-leg posture

the movement is treated as a potential sitting posture rather than a valid squat.

This helps distinguish an actual squat from a chair-sitting movement.

---

# 🔄 EMA Landmark Smoothing

Raw pose landmarks can naturally contain small frame-to-frame movements.

To reduce this jitter, the project applies an **Exponential Moving Average (EMA)** filter.

```text
Smoothed Position =
    α × Current Position
    +
    (1 - α) × Previous Position
```

The default smoothing factor is:

```text
α = 0.4
```

A lower value produces smoother but potentially slower movement, while a higher value makes the system more responsive but less stable.

This smoothing is applied to the tracked shoulder, hip, knee and ankle landmarks.

---

# 🖥️ Live Interface

The application displays the webcam feed with a real-time HUD containing:

```text
┌──────────────────────────────────────┐
│ SQUAT ANALYZER                       │
│                                      │
│ Good Squat!                          │
│ Reps: 8                              │
│ Knee: 92°                            │
│ Hip: 78°                             │
└──────────────────────────────────────┘

                         FPS: 30
```

The interface also displays:

* Pose skeleton
* Joint markers
* Knee angle
* Hip angle
* Rep counter
* Form feedback
* FPS counter
* Keyboard controls

Angle arcs are rendered around the knee and hip joints to visually communicate the measured angles.

---

# 🏗️ Project Architecture

The project is divided into specialized modules:

```text
CV-Squat-Project/
│
├── main.py
│
├── pose_detector.py
│
├── angle_calculator.py
│
├── squat_logic.py
│
├── utils.py
│
├── pose_landmarker_full.task
│
├── pose_landmarker_lite.task
│
├── requirements.txt
│
└── .gitignore
```

---

# 📁 File Descriptions

### `main.py`

The main application entry point.

Responsibilities:

* Initialize the pose detector
* Initialize the squat analyzer
* Open the webcam
* Capture and mirror frames
* Run pose detection
* Extract body landmarks
* Calculate knee and hip angles
* Update squat state
* Render the HUD
* Display FPS
* Handle keyboard controls
* Clean up resources

The application uses a **1280 × 720** camera configuration by default.

---

### `pose_detector.py`

Handles all MediaPipe Pose Landmarker functionality.

Responsibilities:

* Load the `.task` pose model
* Automatically select Full/Lite model
* Run MediaPipe in `VIDEO` mode
* Extract body landmarks
* Apply EMA smoothing
* Convert normalized coordinates into pixel coordinates
* Validate landmark visibility
* Draw the body skeleton
* Release MediaPipe resources

---

### `angle_calculator.py`

Contains the mathematical functions used for joint-angle calculations.

Main functions:

```python
calculate_angle()
get_knee_angle()
get_hip_angle()
```

The calculations use NumPy vector mathematics and return angles in degrees.

---

### `squat_logic.py`

Contains the core squat analysis engine.

Responsibilities:

* Squat state management
* Repetition counting
* Standing confirmation
* Squat depth evaluation
* Back posture evaluation
* Sitting posture rejection
* Live feedback generation
* Form-quality classification

Main class:

```python
SquatAnalyzer
```

---

### `utils.py`

Contains the visual rendering utilities.

Responsibilities:

* HUD panels
* Feedback text
* Angle indicators
* Skeleton colors
* Rep counter display
* FPS display
* Keyboard instructions
* Semi-transparent overlays

---

# 🛠️ Tech Stack

| Technology    | Version  | Purpose                                        |
| ------------- | -------- | ---------------------------------------------- |
| **Python**    | 3.10+    | Core programming language                      |
| **OpenCV**    | ≥ 4.8.0  | Webcam capture, image processing and rendering |
| **MediaPipe** | ≥ 0.10.0 | AI-based human pose detection                  |
| **NumPy**     | ≥ 1.24.0 | Mathematical operations and angle calculations |

The project's dependency requirements are defined in `requirements.txt`.

---

# 📋 Requirements

Before running the project, make sure you have:

* Python **3.10 or higher**
* A working webcam
* Windows, macOS or Linux
* Sufficient lighting for reliable pose detection

A clear view of the full body is recommended for accurate squat analysis.

---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/Vidit0305/CV-Squat-Project.git

cd CV-Squat-Project
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires:

```text
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

# 🤖 Pose Model Setup

The repository contains:

```text
pose_landmarker_full.task
pose_landmarker_lite.task
```

The application automatically searches for the Full model first and uses the Lite model as a fallback.

If the model files are missing, download the MediaPipe Pose Landmarker model and place the `.task` file in the project root.

---

# ▶️ Run the Application

Start the analyzer with:

```bash
python main.py
```

Once launched:

1. The webcam opens.
2. The camera feed is mirrored.
3. MediaPipe detects your body.
4. The skeleton appears on screen.
5. Stand upright to initialize the analyzer.
6. Perform a squat.
7. The system tracks your knee and hip angles.
8. Complete the movement by returning to the standing position.
9. Your repetition is counted.
10. Feedback is displayed in real time.

---

# 🎮 Controls

| Key   | Action                   |
| ----- | ------------------------ |
| **Q** | Quit the application     |
| **R** | Reset repetition counter |

The controls are also displayed at the bottom of the application window.

---

# 📐 Example Squat Cycle

A typical valid repetition follows:

```text
        STANDING
           │
           │
           ▼
      Begin Squat
           │
           ▼
     Knee < 120°
           │
           ▼
       SQUATTING
           │
           ├── Check Depth
           │
           ├── Check Back
           │
           └── Check Posture
           │
           ▼
     Knee > 155°
           │
           ▼
        STANDING
           │
           ▼
       REP + 1
```

---

# ⚙️ Configuration

Several parameters can be modified directly in the source code.

### Camera Configuration

Located in `main.py`:

```python
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
```

### Pose Detection

Located in `main.py`:

```python
min_detection_confidence=0.7
min_tracking_confidence=0.7
smoothing_factor=0.4
```

### Squat Thresholds

Located in `squat_logic.py`:

```python
KNEE_ANGLE_SQUAT = 120
KNEE_ANGLE_STAND = 155
KNEE_ANGLE_DEEP = 100

HIP_ANGLE_MIN = 55
HIP_ANGLE_GOOD_MIN = 65
```

These values can be tuned depending on camera position, body proportions and desired form criteria.

---

# 🔬 Technical Highlights

This project demonstrates several important computer vision concepts:

### 1. Human Pose Estimation

MediaPipe detects body landmarks from a normal RGB webcam feed.

### 2. Landmark Smoothing

EMA filtering is used to stabilize noisy landmark coordinates.

### 3. Geometric Analysis

Joint angles are calculated using vector mathematics and dot products.

### 4. Temporal State Machine

Squat repetitions are detected based on movement transitions rather than a single frame.

### 5. Form Analysis

The system evaluates:

* Squat depth
* Back posture
* Standing posture
* Ankle/hip alignment

### 6. Real-Time Computer Vision

The entire pipeline runs continuously on live webcam frames.

---

# ⚡ Performance Considerations

The application is designed for real-time processing.

Key performance techniques include:

* MediaPipe `VIDEO` running mode
* Temporal landmark tracking
* Single-person pose detection
* EMA smoothing
* Lightweight OpenCV rendering
* Direct frame processing
* Averaged bilateral landmarks

MediaPipe's `VIDEO` mode allows the pose detector to operate with temporal information between frames rather than treating every frame as an entirely independent image.

---

# 🔧 Troubleshooting

| Problem                                | Possible Solution                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Camera does not open**               | Check camera permissions and make sure another application is not using it                   |
| **Person is not detected**             | Improve lighting and ensure your full body is visible                                        |
| **Landmarks are unstable**             | Move slightly farther from the camera and improve lighting                                   |
| **Reps are not counting**              | Start from a clearly upright standing position                                               |
| **Squat detected as sitting**          | Keep your ankles approximately underneath your hips                                          |
| **Feedback says "Go Lower"**           | Increase squat depth                                                                         |
| **Feedback says "Keep Back Straight"** | Reduce excessive forward torso bending                                                       |
| **Low FPS**                            | Reduce camera resolution or use the Lite pose model                                          |
| **Model not found**                    | Ensure `pose_landmarker_full.task` or `pose_landmarker_lite.task` exists in the project root |

---

# 🚧 Current Limitations

The current implementation is intentionally focused on single-person squat analysis.

Potential limitations include:

* Primarily designed for a single person
* Uses 2D image coordinates for geometric analysis
* Camera angle can affect measured angles
* Poor lighting can reduce pose detection quality
* Full-body visibility is important
* Thresholds may need adjustment for different users and camera setups
* The system is an assistive computer-vision analyzer and should not be treated as professional medical or fitness advice

---

# 🔮 Future Enhancements

Potential improvements include:

* [ ] Support multiple exercises
* [ ] Add lunges and jumping-jack detection
* [ ] Add push-up analysis
* [ ] Add personalized form thresholds
* [ ] Add workout session tracking
* [ ] Add calories-burn estimation
* [ ] Add exercise history
* [ ] Add performance graphs
* [ ] Add voice feedback
* [ ] Add rep tempo analysis
* [ ] Add left/right leg symmetry analysis
* [ ] Add automatic exercise recognition
* [ ] Add session summaries
* [ ] Build a desktop GUI
* [ ] Export workout statistics
* [ ] Add configurable camera calibration
* [ ] Add multi-person pose analysis

---

# 📚 Learning Outcomes

Building this project provides practical experience with:

* Computer Vision
* Human Pose Estimation
* MediaPipe Tasks API
* OpenCV
* Real-Time Video Processing
* Landmark Detection
* Geometric Mathematics
* Vector Operations
* State Machines
* Temporal Analysis
* Exponential Moving Average Filtering
* Real-Time UI Rendering
* Python Modular Architecture

---

# 📄 License

This project is open-source and available under the **MIT License**.

See the `LICENSE` file for more information.

---

<div align="center">

## 🏋️ Built with Computer Vision & AI

**Made by Vidit Sharma**

*Turning a webcam into an intelligent squat form analyzer.*

⭐ **If you found this project interesting, consider giving the repository a star!**

</div>
