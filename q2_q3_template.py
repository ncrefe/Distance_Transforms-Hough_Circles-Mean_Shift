"""
Task 2: Hough Transform for Circle Detection
Task 3: Mean Shift for Peak Detection in Hough Accumulator
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    max_radius = min(max_radius, int(np.linalg.norm(edges.shape)))

    edges_points = np.array(np.nonzero(edges))
    h, w = edges.shape

    # Define radii and theta steps
    # radii: list of possible circle radii to test (in pixels)
    #   -> r_ssz = 1 means we check every 1 pixel radius (more precise but slower)
    # thetas: angle values (0–360°) used to find possible circle centers around each edge point
    #   -> theta_ssz controls the angular step (smaller = more accurate, slower)
    radii = np.arange(min_radius, max_radius + 1, r_ssz)
    thetas = np.deg2rad(np.arange(0, 360, theta_ssz))

    # Create accumulator (r x h x w)
    # 3D accumulator: stores votes for all possible circles (r, y, x). Because we have three unknown parameters .
    # Each edge point votes for possible circle centers at each radius.
    # The cell with the highest votes corresponds to the most likely circle.
    accumulator = np.zeros((len(radii), h, w), dtype=np.uint32)

    # Iterate over all edge points
    for y0, x0 in zip(edges_points[0], edges_points[1]):
        for r_idx, r in enumerate(radii):
            # For each theta, compute possible center
            a = x0 - (r * np.cos(thetas))
            b = y0 - (r * np.sin(thetas))
            a = np.round(a).astype(int)
            b = np.round(b).astype(int)
            # Increment accumulator if within bounds
            valid = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            accumulator[r_idx, b[valid], a[valid]] += 1

    # Detect circles above threshold
    # Example for detected_circles, tuple sample:
    # (np.int64(280), np.int64(146), np.int64(31), np.uint32(122))
    # Center: (280, 146)
    # Radius: 31 px
    # Vote Count: 122
    detected_circles = []
    for r_idx, r in enumerate(radii):
        acc_slice = accumulator[r_idx]
        # Find local maxima above threshold
        ys, xs = np.nonzero(acc_slice >= threshold)
        votes = acc_slice[ys, xs]
        for y, x, v in zip(ys, xs, votes):
            # Check min_dist to avoid duplicates
            if all(np.linalg.norm(np.array([x, y]) - np.array([cx, cy])) >= min_dist
                   for cx, cy, _, _ in detected_circles):
                detected_circles.append((x, y, r, v))

    return detected_circles, accumulator


def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift
        threshold: Minimum value to consider (if None, use fraction of max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape

    # Threshold: ignore low votes
    if threshold is None:
        threshold = accumulator.max() * 0.5  # use half of max by default

    peaks = []
    visited = np.zeros_like(accumulator, dtype=bool)

    # Iterate over all cells above threshold
    r_idxs, ys, xs = np.nonzero(accumulator >= threshold)

    for r_idx, y, x in zip(r_idxs, ys, xs):
        if visited[r_idx, y, x]:
            continue

        # Initialize mean shift
        shift_x, shift_y = x, y
        while True:
            # Define neighborhood
            y_min = max(0, shift_y - bandwidth)
            y_max = min(h, shift_y + bandwidth + 1)
            x_min = max(0, shift_x - bandwidth)
            x_max = min(w, shift_x + bandwidth + 1)

            # Extract neighborhood values
            patch = accumulator[r_idx, y_min:y_max, x_min:x_max]
            ys_patch, xs_patch = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')

            # Compute weighted mean of neighbors
            total_weight = patch.sum()
            if total_weight == 0:
                break
            new_shift_y = int(round((ys_patch * patch).sum() / total_weight))
            new_shift_x = int(round((xs_patch * patch).sum() / total_weight))

            # Check convergence
            if new_shift_y == shift_y and new_shift_x == shift_x:
                break
            shift_y, shift_x = new_shift_y, new_shift_x

        # Mark the neighborhood around the found peak as visited
        # This ensures that all cells within the bandwidth around the peak are not processed again.
        # The purpose is to avoid detecting duplicate peaks that are very close to each other.
        # Essentially, once a local maximum is found, its surrounding patch is "blocked" from further consideration.
        y_min = max(0, shift_y - bandwidth)
        y_max = min(h, shift_y + bandwidth + 1)
        x_min = max(0, shift_x - bandwidth)
        x_max = min(w, shift_x + bandwidth + 1)
        visited[r_idx, y_min:y_max, x_min:x_max] = True

        # Save peak
        # When performing mean shift on the accumulator:
        # For example, if we start at position (1, 3) with a bandwidth of 10,
        # the algorithm looks at all pixels within that bandwidth (a neighborhood)
        # and computes the weighted mean position based on accumulator values.
        # As the shift moves towards the local maximum (the peak), it may visit
        # many pixels along the path, say 200 pixels in total.
        # All of these visited pixels are considered part of the same peak,
        # because they converge to the same maximum point in the accumulator.
        # Once convergence is reached, the neighborhood around the peak is marked
        # as visited so that subsequent searches do not consider these pixels again.
        # Therefore, even though 200 pixels were involved in the mean shift computation,
        # they all "belong" to the same detected peak.
        peaks.append((shift_x, shift_y, r_idx, accumulator[r_idx, shift_y, shift_x]))

    return peaks


def main():
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)

    img_path = 'data/coins.jpg'

    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return

    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    min_radius = 20
    max_radius = 60
    threshold = 120  # minimum votes in accumulator to be considered a circle
    min_dist = 30  # minimum distance between circle centers
    r_ssz = 1  # radius step size in pixels
    theta_ssz = 1  # angular step size in degrees

    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for x, y, r, v in detected_circles:
        circ = Circle((x, y), r, color='red', fill=False, linewidth=2)
        ax.add_patch(circ)
    ax.set_title("Detected Circles using Hough Transform")
    plt.show()

    # Visualize accumulator slices
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    r_indices = np.linspace(0, len(accumulator) - 1, 9, dtype=int)
    for ax, r_idx in zip(axes.flatten(), r_indices):
        ax.imshow(accumulator[r_idx], cmap='hot')
        ax.set_title(f'Radius index: {r_idx}')
    plt.tight_layout()
    plt.show()

    # Visualize peak radius
    total_votes = accumulator.max(axis=(1, 2))
    peak_r_idx = np.argmax(total_votes)
    plt.imshow(accumulator[peak_r_idx], cmap='hot')
    plt.title(f"Accumulator slice at peak radius index {peak_r_idx}")
    plt.colorbar()
    plt.show()

    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print("  - Canny thresholds affect edge quality and thus detection")

    # Statistics from output image:
    print("Statistics for different Canny thresholds:")
    print("Canny=(75,150) → Detected=10 → Good balance, captures most coins")
    print("Canny=(100,200) → Detected=9  → Slightly stricter, loses 1 coin")
    print("Canny=(150,250) → Detected=3  → Too strict, misses most coins")

    print("\nComment:")
    print(
        "Canny thresholds directly control edge quality fed into Hough transform.\n"
        "Lower thresholds (e.g., 75,150) preserve more edges — including faint ones — leading to better circle detection.\n"
        "Higher thresholds (e.g., 150,250) remove too many edges, especially on small or low-contrast coins,\n"
        "causing Hough to miss circles due to insufficient votes.\n"
        "Optimal Canny range depends on image contrast — here, (75,150) gives best results."
    )

    print("\n" + "*" * 70)
    # Statistics from output image:
    print("Statistics for different thresholds:")
    print("threshold=50  → Detected=14 → Over-detection, noisy, overlapping circles")
    print("threshold=140 → Detected=4  → Under-detection, misses small/weak circles")
    print("threshold=170 → Detected=1  → Too strict, only strongest circle remains")

    print("\nComment:")
    print(
        "The threshold parameter controls the minimum votes required to accept a circle.\n"
        "Low threshold (e.g., 50) is sensitive and detects many false positives —\n"
        "because weak edges or noise can accumulate enough votes to pass the low bar.\n"
        "High threshold (e.g., 170) is too strict — real circles with incomplete or faint edges\n"
        "fail to reach the vote count, even if they are visually clear.\n"
        "Medium values (e.g., 140) strike a balance but may still miss some coins.\n"
        "Optimal threshold depends on edge quality and noise — here, ~120–140 gives best trade-off."
    )

    print("\n" + "*" * 70)
    print("Statistics for different min_radius:")
    print("min_radius=15 → Detected=9  → Detects small coins, may include noise")
    print("min_radius=50 → Detected=3  → Only large coins detected, misses small ones")
    print("min_radius=60 → Detected=0  → No circles found — too high for any coin")

    print("\nComment:")
    print(
        "The min_radius parameter sets the smallest circle size to detect.\n"
        "Too low (e.g., 15) allows small noisy shapes or tiny coins to be detected.\n"
        "Too high (e.g., 60) excludes all real coins in the image, resulting in zero detections.\n"
        "Medium values (e.g., 20–40) are optimal for this image, capturing most coins without noise."
    )

    print("\n" + "*" * 70)
    print("Statistics for different max_radius:")
    print("max_radius=30 → Detected=0  → Too small, excludes all real coins")
    print("max_radius=40 → Detected=4  → Captures medium/large coins, misses small ones")
    print("max_radius=80 → Detected=9  → Fully covers all coin sizes, optimal for this image")

    print("\nComment:")
    print(
        "The max_radius parameter defines the largest circle size to search for.\n"
        "Too small (e.g., 30) fails to detect any coin — too restrictive.\n"
        "Medium values (e.g., 40) miss smaller coins but capture larger ones.\n"
        "Larger values (e.g., 80) include all coins without over-detection, making it ideal here."
    )

    print("\n" + "*" * 70)
    print("Statistics for different r_ssz:")
    print("r_ssz=0.5 → Detected=9  → High precision, captures all coins accurately")
    print("r_ssz=5   → Detected=4  → Coarse search, misses small/medium coins")
    print("r_ssz=10  → Detected=3  → Very coarse, only largest coins detected")

    print("\nComment:")
    print(
        "The r_ssz parameter controls the step size in radius space during Hough voting.\n"
        "Small values (e.g., 0.5) are precise but computationally expensive — detect all coins.\n"
        "Larger values (e.g., 5–10) speed up computation but reduce accuracy — miss smaller circles\n"
        "because the true radius may fall between sampled steps and receive no votes.\n"
        "Only the largest coins survive because their strong edges accumulate enough votes\n"
        "even at mismatched radii, while smaller coins lack sufficient support."
    )

    print("\n" + "*" * 70)
    print("Statistics for different theta_ssz:")
    print("theta_ssz=0.2 → Detected=9  → Fine angular sampling, captures all coins accurately")
    print("theta_ssz=1.5 → Detected=4  → Moderate sampling, misses small/weak circles")
    print("theta_ssz=3   → Detected=0  → Too coarse, no circle detected — insufficient coverage")

    print("\nComment:")
    print(
        "The theta_ssz parameter controls the angular step size when computing possible circle centers.\n"
        "Small values (e.g., 0.2) provide dense sampling, ensuring accurate center voting for all coins.\n"
        "Larger steps (e.g., 3) skip too many angles — edge points fail to vote consistently for true centers,\n"
        "causing votes to scatter and never reach the detection threshold.\n"
        "Only very strong circles might survive at moderate steps (e.g., 1.5), but fine resolution (≤1°) is essential for reliable detection."
    )

    print("\n")

    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)

    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    print("Applying mean shift to find peaks...")
    bandwidth = 10
    peaks = myMeanShift(accumulator, bandwidth, threshold=threshold)

    # Visualize corresponding circles on original image    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for x, y, r_idx, v in peaks:
        r = min_radius + r_idx * r_ssz  # convert r_idx back to actual radius
        circ = Circle((x, y), r, color='blue', fill=False, linewidth=2)
        ax.add_patch(circ)
    ax.set_title("Detected Circles using Mean Shift Peaks")
    plt.show()

    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")

    # Statistics from output image:
    print("Statistics:")
    print("Bandwidth=1 → Peaks=20  → Over-detection, noisy, overlapping circles")
    print("Bandwidth=5 → Peaks=16  → Clean, accurate, minimal noise")
    print("Bandwidth=30→ Peaks=16  → Slight merging, acceptable")
    print("Bandwidth=60→ Peaks=16  → Severe merging, duplicate circles")

    print("\nComment:")
    print(
        "The bandwidth parameter in mean shift directly affects peak detection sensitivity.\n"
        "Small bandwidth (e.g., 1) is overly sensitive to noise and produces many false peaks.\n"
        "Medium values (e.g., 5) clearly detect real circles while suppressing noise.\n"
        "However, very large bandwidth (e.g., 60) merges distinct circles into single peaks\n"
        "and may draw the same circle multiple times.\n"
        "Thus, medium values like bandwidth=5 provide the best balance between precision and noise robustness."
    )

    print("=" * 70)
    print("Task 3 complete!")


def parameter_sweep_hough_transform(edges, myHoughCircles):
    # Fixed setup
    setup = {
        "min_radius": 20,
        "max_radius": 60,
        "threshold": 120,
        "min_dist": 30,
        "r_ssz": 1,
        "theta_ssz": 1
    }

    # Parameter lists for sweep
    param_values = {
        "threshold": [50, 140, 170],
        "min_radius": [15, 50, 60],
        "max_radius": [30, 40, 80],
        "r_ssz": [0.5, 5, 10],
        "theta_ssz": [0.2, 1.5, 3]
    }

    # Sweep each parameter
    for param, values in param_values.items():
        fig, axes = plt.subplots(1, len(values), figsize=(15, 5))
        for ax, val in zip(axes, values):
            # Prepare parameters: update the current param, keep others fixed
            params = setup.copy()
            params[param] = val

            # Run Hough Transform
            detected_circles, accumulator = myHoughCircles(
                edges,
                params["min_radius"],
                params["max_radius"],
                params["threshold"],
                params["min_dist"],
                params["r_ssz"],
                params["theta_ssz"]
            )

            # Draw circles
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for x, y, r, v in detected_circles:
                circ = Circle((x, y), r, color='red', fill=False, linewidth=2)
                ax.add_patch(circ)
            ax.set_title(f"{param}={val}\nDetected={len(detected_circles)}")
            ax.axis('off')

        plt.suptitle(f"Effect of {param} on Circle Detection")
        plt.tight_layout()
        plt.show()


def parameter_sweep_mean_shift(edges, myHoughCircles, myMeanShift):
    # Fixed Hough setup
    hough_setup = {
        "min_radius": 20,
        "max_radius": 60,
        "threshold": 120,
        "min_dist": 30,
        "r_ssz": 1,
        "theta_ssz": 1
    }

    # Bandwidth values to test
    bandwidth_values = [1, 5, 30, 60]

    # Run Hough Transform once (fixed parameters)
    detected_circles, accumulator = myHoughCircles(
        edges,
        hough_setup["min_radius"],
        hough_setup["max_radius"],
        hough_setup["threshold"],
        hough_setup["min_dist"],
        hough_setup["r_ssz"],
        hough_setup["theta_ssz"]
    )

    # Sweep over bandwidth values
    fig, axes = plt.subplots(1, len(bandwidth_values), figsize=(20, 5))
    for ax, bw in zip(axes, bandwidth_values):
        peaks = myMeanShift(accumulator, bandwidth=bw, threshold=hough_setup["threshold"])

        # Visualize circles corresponding to peaks
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for x, y, r_idx, v in peaks:
            r = hough_setup["min_radius"] + r_idx * hough_setup["r_ssz"]
            circ = Circle((x, y), r, color='blue', fill=False, linewidth=2)
            ax.add_patch(circ)

        ax.set_title(f"Bandwidth={bw}\nPeaks={len(peaks)}")
        ax.axis('off')

    plt.suptitle("Effect of Bandwidth on Mean Shift Peak Detection")
    plt.tight_layout()
    plt.show()


def parameter_sweep_canny_thresholds(gray, myHoughCircles):
    """
    Test the effect of different Canny thresholds on circle detection results.
    """

    # -----------------------------
    # Fixed setup for Hough Transform
    # -----------------------------
    setup = {
        "min_radius": 20,
        "max_radius": 60,
        "threshold": 120,
        "min_dist": 30,
        "r_ssz": 1,
        "theta_ssz": 1
    }

    # -----------------------------
    # Define Canny threshold pairs to test
    # -----------------------------
    canny_pairs = [
        (75, 150),
        (100, 200),
        (150, 250)
    ]

    # -----------------------------
    # Run tests
    # -----------------------------
    fig, axes = plt.subplots(1, len(canny_pairs), figsize=(18, 6))
    for ax, (low, high) in zip(axes, canny_pairs):
        # Edge detection
        edges = cv2.Canny(gray, low, high)

        # Detect circles
        detected_circles, _ = myHoughCircles(
            edges,
            setup["min_radius"],
            setup["max_radius"],
            setup["threshold"],
            setup["min_dist"],
            setup["r_ssz"],
            setup["theta_ssz"]
        )

        # Draw circles
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for x, y, r, v in detected_circles:
            circ = Circle((x, y), r, color='red', fill=False, linewidth=2)
            ax.add_patch(circ)
        ax.set_title(f"Canny=({low},{high})\nDetected={len(detected_circles)}")
        ax.axis("off")

    plt.suptitle("Effect of Canny Thresholds on Hough Circle Detection", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    # Path to the image
    img_path = 'data/coins.jpg'

    # Load image and convert to grayscale
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: {img_path} not found!")
        assert "Image not found" in img_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    # === Additional analysis functions for parameter tuning ===

    # parameter_sweep_hough_transform: Systematically tests different Hough Transform parameters (e.g., threshold, radius range, step sizes) and visualizes how they affect circle detection results.
    # parameter_sweep_hough_transform(edges, myHoughCircles)

    # parameter_sweep_mean_shift: Applies Mean Shift on the Hough accumulator with varying bandwidth values to analyze how peak detection sensitivity changes.
    # parameter_sweep_mean_shift(edges, myHoughCircles, myMeanShift)

    # parameter_sweep_canny_thresholds: Tests multiple Canny edge threshold pairs (low/high) to evaluate their impact on edge quality and final circle detection.
    # parameter_sweep_canny_thresholds(gray, myHoughCircles)
