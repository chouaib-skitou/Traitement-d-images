import cv2
import numpy as np


image = cv2.imread("Michelangelo_ThecreationofAdam_1707x775.jpg")

(h, w) = image.shape[:2] # on récupère la hauteur et la largeur
center = (w / 2, h / 2) # on en déduit la position du centre de l'image

print(h)
print(w)

img = np.zeros((h, w, 3), np.uint8)

# Display the image
cv2.imshow("Binary", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
