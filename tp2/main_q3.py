import cv2
import numpy as np
import math

def Sobel_gradient(image, N=5, rmin=1, rmax=100, delta_r=2, cmin=1, cmax=100, delta_c=2, radmin=2, radmax=30, delta_rad=2):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  

    gradient_direction = np.arctan2(sobel_y, sobel_x)  # Direction en radians
    # valeurs entre 0 et 2pi
    gradient_direction = np.mod(gradient_direction, 2 * np.pi)

    acc_shape = (
        (rmax - rmin) // delta_r + 1,
        (cmax - cmin) // delta_c + 1,
        (radmax - radmin) // delta_rad + 1,
    )
    accumulator = np.zeros(acc_shape, dtype=int)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    threshold_ratio = 0.25  # Seuillage à 25% de la valeur maximale de magnitude
    threshold = threshold_ratio * magnitude.max()
    contour_pixels = (magnitude > threshold).astype(np.uint8)

    # vote seulement dans la direction du gradient
    contour_indices = np.argwhere(contour_pixels)
    for y, x in contour_indices:
        # Calculer la direction du gradient pour ce pixel
        direction = gradient_direction[y, x]

        # incrémenter que les cases proches de la direction du gradient
        for r_idx, r in enumerate(range(rmin, rmax + 1, delta_r)):
            for c_idx, c in enumerate(range(cmin, cmax + 1, delta_c)):
                rad = int(np.sqrt((x - c) ** 2 + (y - r) ** 2))

                if radmin <= rad <= radmax:
                    rad_idx = (rad - radmin) // delta_rad
                    # l'angle entre la direction du gradient et l'angle du cercle
                    circle_angle = np.arctan2(y - r, x - c)
                    angle_diff = np.abs(np.mod(circle_angle - direction, 2 * np.pi))  # Différence d'angle

                    angle_tolerance = np.pi / 6  # Tolérance de 30 degrés
                    if angle_diff <= angle_tolerance:
                        accumulator[r_idx, c_idx, rad_idx] += magnitude[y, x] / rad  # Pondération
