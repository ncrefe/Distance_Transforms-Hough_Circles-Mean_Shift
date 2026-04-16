"""
Task 1: Distance Transform using Chamfer 5-7-11
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
    
    Based on Borgefors "Distance transformations in digital images" (1986).
    
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are 255, background is 0
    
    Returns:
        Distance transform image
    """
    h, w = binary_image.shape
    dt = np.full((h, w), np.inf, dtype=np.float32)
    
    # Initialize: 0 if feature pixel, infinity otherwise
    dt[binary_image > 0] = 0
    
    # Define forward and backward masks with (row_offset, col_offset, distance)
    # Forward mask (as shown in slide 37)
    center = (h//2, w//2)
    # How we encode this: 
    # (delta_y, delta_x, weight of the pixel (center[0]+delta_y,center[1]+delta_x))
    forward_mask = [(-1, -2, 11), (-1, -1, 7), (-1, 0, 5), (0, -1, 5), (-1, 1, 7), (-1, 2, 11)]

    # Backward mask (as shown in slide 37)
    # Same as the forward mask however we have to adjust to increase the indicies => thus no minus signs
    backward_mask = [(1, 2, 11), (1, 1, 7), (1, 0, 5), (0, 1, 5), (1, -1, 7), (1, -2, 11)]
    
    # Forward pass
    for i in range(h-1):
        for j in range(w-1): 
            for d_i, d_j, w_ij in forward_mask: 
                # Calculate the neighboor pixel like explained above
                n_i = d_i + i 
                n_j = d_j + j

                # Sanity check 
                if 0 <= n_i <= h-1 and 0<= n_j <= w-1: 
                    dt[i,j] = min(dt[i,j],dt[n_i, n_j]+w_ij)
    
    # Backward pass 
    # Symmetric to Forward 
    for i in range(h-1,-1,-1):
        for j in range(w-1,-1,-1): 
            for d_i, d_j, w_ij in backward_mask: 
                n_i = d_i + i 
                n_j = d_j + j

                # Sanity chcek 
                if 0 <= n_i <= h-1 and 0<= n_j <= w-1: 
                    dt[i, j] = min(dt[i, j], dt[n_i, n_j] + w_ij)
    
    return dt


def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    #img_path = 'data/bonn.jpg'
    img_path = 'data/circle.png'      # play with different images
    # img_path = 'data/square.png'      
    # img_path = 'data/triangle.png'    
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    # Define Threshold Values 
    T_LOW = 50 
    T_HIGH = 150 
    binary_img = cv2.Canny(img, T_LOW, T_HIGH)
    
    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    dist_transform = chamfer_distance_transform_5_7_11(binary_img)

    # Compute distance transform using cv2.distanceTransform
    binary_inv = cv2.bitwise_not(binary_img)
    cv2_dist_transform = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 5)
    cv2_dist_transform = cv2.normalize(cv2_dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO
    # 1. Original image
    # 2. Edge image
    # 3. Distance transform
    # 4. Distance transform using OpenCV
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')

    # 2. Edge image
    axes[0, 1].imshow(binary_img, cmap='gray')
    axes[0, 1].set_title('Edge Image')

    # 3. Distance transform (your own implementation)
    axes[1, 0].imshow(dist_transform, cmap='hot')
    axes[1, 0].set_title('Chamfer Distance Transform (Ours)')

    # 4. Distance transform using OpenCV
    axes[1, 1].imshow(cv2_dist_transform, cmap='hot')
    axes[1, 1].set_title('Distance Transform (OpenCV)')

    plt.show()

    print("\n" + "=" * 70)
    print("Task 1 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
    