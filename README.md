# Distance Transforms, Hough Circles & Mean Shift

This project implements classical computer vision algorithms from scratch and compares them with built-in OpenCV methods. The focus is on understanding **distance transforms, shape detection, and peak finding in accumulator spaces**.

All implementations are compatible with **Linux, Python 3.12, OpenCV 4.11, and NumPy 2.3.3**.

---

## 🚀 Features

* 🧭 Chamfer Distance Transform (5-7-11 metric)
* 🧱 Two-pass distance transform algorithm
* 🎯 Circle detection using Hough Transform (custom implementation)
* 📊 Accumulator visualization across radii
* 📍 Peak detection using Mean Shift
* ⚖️ Comparison with OpenCV built-in distance transform
* 🖼️ Visualization of edges, transforms, and detected circles

---

## 🛠️ Tech Stack

* Python 3.12
* OpenCV 4.11
* NumPy 2.3.3
* matplotlib

> ⚠️ Only the allowed libraries are used
> ⚠️ Designed for Linux-based execution

---

## 📂 Project Structure

```
.
├── uni-bonn.jpg
├── coins.jpg
├── q1_distance_transform.py
├── q2_hough_circles.py
├── q3_mean_shift.py
├── README.md
```

---

## ⚙️ Installation

```
pip install opencv-python numpy matplotlib
```

---

## ▶️ Usage

Run tasks individually:

```
python q1_distance_transform.py
python q2_hough_circles.py
python q3_mean_shift.py
```

---

## 🧪 Implemented Tasks

### 1. Chamfer Distance Transform (5-7-11)

* Apply Canny edge detection using OpenCV
* Implement Chamfer 5-7-11 distance transform using a two-pass algorithm
* Compute distance to nearest edge pixel
* Compare with `cv2.distanceTransform`

---

### 2. Hough Transform for Circle Detection

* Implement `myHoughCircles` from scratch
* Detect circles in `coins.jpg`

#### 📊 Accumulator Analysis

* Visualize accumulator slices for different radii
* Identify radius with maximum votes

#### ⚙️ Parameter Study

* Effect of:

  * Threshold values
  * Radius range
  * Accumulator resolution

---

### 3. Mean Shift on Hough Accumulator

* Implement `myMeanShift`
* Apply it to Hough accumulator peaks
* Detect stable circle centers

#### 📉 Analysis

* Bandwidth effect on:

  * Number of detected peaks
  * Cluster stability
  * Over/under-detection behavior

---

## 📌 Notes

* Only low-level OpenCV functions are used (Canny allowed)
* Focus is on geometric voting and iterative optimization methods
* Results are validated visually and through parameter analysis
