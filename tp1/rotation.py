import cv2
import numpy as np

def analyser_fichier(fichier, index_recherche):
    # Ouvrir le fichier et lire son contenu
    with open(fichier, 'r') as f:
        lignes = f.readlines()

    # Parcourir chaque ligne du fichier
    for ligne in lignes:
        # Diviser la ligne en utilisant les espaces comme séparateurs
        elements = ligne.split()
        
        # Vérifier si la ligne a bien 4 éléments (index, posx, posy, angle)
        if len(elements) == 4:
            # Extraire l'index, posx, posy, et angle
            index = int(elements[0])
            angle = float(elements[3])
            
            # Vérifier si l'index correspond à celui recherché
            if index == index_recherche:
                return angle


num = 24
image = cv2.imread("frag_eroded/frag_eroded_" + str(num) + ".png")

(h, w) = image.shape[:2] # on récupère la hauteur et la largeur
center = (w / 2, h / 2) # on en déduit la position du centre de l'image
angle = analyser_fichier("fragments.txt",24) #-88.581 # on indique l'angle de rotation voulue
scale = 1 # on précise l'échelle de l'image

M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(image, M, (w, h))

cv2.imshow("original Image", image) # on affiche l'image d'origine
cv2.imshow("Rotated Image", rotated) # on affiche l'image retourner
cv2.waitKey(0)
cv2.destroyAllWindows()