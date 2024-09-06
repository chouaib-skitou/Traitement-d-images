import cv2
import numpy as np


num = 24
image = cv2.imread("frag_eroded/frag_eroded_" + str(num) + ".png")

(h, w) = image.shape[:2] # on récupère la hauteur et la largeur
center = (w / 2, h / 2) # on en déduit la position du centre de l'image
angle = -88.581 # on indique l'angle de rotation voulue
scale = 1 # on précise l'échelle de l'image

M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(image, M, (w, h))

cv2.imshow("original Image", image) # on affiche l'image d'origine
cv2.imshow("Rotated Image", rotated) # on affiche l'image retourner
cv2.waitKey(0)
cv2.destroyAllWindows()