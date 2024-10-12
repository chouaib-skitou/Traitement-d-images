import cv2 
import numpy as np

def inserer_fragment(image, fragment, decalage_x, decalage_y, angle):
    hauteur_frag, largeur_frag = fragment.shape[:2]

    decalage_x = decalage_x - (hauteur_frag // 2)
    decalage_y = decalage_y - (largeur_frag // 2)

    centre = (largeur_frag / 2, hauteur_frag / 2)
    matrice_rotation = cv2.getRotationMatrix2D(centre, angle, 1)

    fragment_tourne = cv2.warpAffine(
        src=fragment, M=matrice_rotation, dsize=(largeur_frag, hauteur_frag))

    fragment_bgr = fragment_tourne[:, :, :3]
    fragment_alpha = fragment_tourne[:, :, 3] / 255.0

    fin_y = min(decalage_y + hauteur_frag, image.shape[0])
    fin_x = min(decalage_x + largeur_frag, image.shape[1])

    debut_frag_y = 0
    debut_frag_x = 0

    if decalage_y < 0:
        debut_frag_y = -decalage_y
        decalage_y = 0
    if decalage_x < 0:
        debut_frag_x = -decalage_x
        decalage_x = 0

    fin_frag_y = debut_frag_y + (fin_y - decalage_y)
    fin_frag_x = debut_frag_x + (fin_x - decalage_x)

    zone_interet = image[decalage_y:fin_y, decalage_x:fin_x]

    for c in range(0, 3):
        zone_interet[:, :, c] = (fragment_alpha[debut_frag_y:fin_frag_y, debut_frag_x:fin_frag_x] * fragment_bgr[debut_frag_y:fin_frag_y, debut_frag_x:fin_frag_x, c] +
                                 (1 - fragment_alpha[debut_frag_y:fin_frag_y, debut_frag_x:fin_frag_x]) * zone_interet[:, :, c])

    image[decalage_y:fin_y, decalage_x:fin_x] = zone_interet



cv2.imwrite('white_img.jpg', np.ones((775, 1707, 3), np.uint8) * 255)

image = cv2.imread('white_img.jpg')
with open('fragments_s.txt', 'r') as fichier:
    indices_ignores = {int(ligne.strip()) for ligne in fichier.readlines()}

with open('fragments.txt', 'r') as fichier:
    lignes = fichier.readlines()

for ligne in lignes:
    elements = ligne.split()

    indice = int(elements[0])
    x = int(elements[1])
    y = int(elements[2])
    angle = float(elements[3])

    if indice in indices_ignores:
        continue
    fragment = cv2.imread(f'frag_eroded/frag_eroded_{int(indice)}.png', cv2.IMREAD_UNCHANGED)
    inserer_fragment(image, fragment, x, y, angle)

cv2.imwrite('image_reconstruite.jpg', image)
cv2.imshow('Image apres reconstruction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
