import cv2 
import numpy as np

def inserer_fragment(image, fragment, decalage_x, decalage_y, angle):
    # Détermination de la taille du fragment et ajustement des décalages pour centrer le fragment
    hauteur_frag, largeur_frag = fragment.shape[:2]
    decalage_x = decalage_x - (hauteur_frag // 2)
    decalage_y = decalage_y - (largeur_frag // 2)

    # Création de la matrice de rotation pour faire pivoter le fragment autour de son centre
    centre = (largeur_frag / 2, hauteur_frag / 2)
    matrice_rotation = cv2.getRotationMatrix2D(centre, angle, 1)

    # Application de la rotation au fragment
    fragment_tourne = cv2.warpAffine(
        src=fragment, M=matrice_rotation, dsize=(largeur_frag, hauteur_frag))

    # Séparation des canaux de couleur et du canal alpha pour la fusion
    fragment_bgr = fragment_tourne[:, :, :3]
    fragment_alpha = fragment_tourne[:, :, 3] / 255.0  # Normalisation de l'alpha

    # Détermination des coordonnées de la zone d'insertion dans l'image principale
    fin_y = min(decalage_y + hauteur_frag, image.shape[0])
    fin_x = min(decalage_x + largeur_frag, image.shape[1])

    debut_frag_y = 0
    debut_frag_x = 0

    # Ajustement des coordonnées de début si elles sont négatives
    if decalage_y < 0:
        debut_frag_y = -decalage_y
        decalage_y = 0
    if decalage_x < 0:
        debut_frag_x = -decalage_x
        decalage_x = 0

    # Calcul des coordonnées de fin pour le fragment dans la zone d'intérêt
    fin_frag_y = debut_frag_y + (fin_y - decalage_y)
    fin_frag_x = debut_frag_x + (fin_x - decalage_x)

    # Sélection de la zone d'intérêt sur l'image où le fragment sera inséré
    zone_interet = image[decalage_y:fin_y, decalage_x:fin_x]

    # Fusion du fragment avec l'image en tenant compte de la transparence (canal alpha)
    for c in range(0, 3):  # Pour chaque canal de couleur
        zone_interet[:, :, c] = (fragment_alpha[debut_frag_y:fin_frag_y, debut_frag_x:fin_frag_x] * fragment_bgr[debut_frag_y:fin_frag_y, debut_frag_x:fin_frag_x, c] +
                                 (1 - fragment_alpha[debut_frag_y:fin_frag_y, debut_frag_x:fin_frag_x]) * zone_interet[:, :, c])

    # Mise à jour de l'image principale avec la zone d'intérêt modifiée
    image[decalage_y:fin_y, decalage_x:fin_x] = zone_interet

# Création d'une image blanche de 775x1707 pixels
cv2.imwrite('white_img.jpg', np.ones((775, 1707, 3), np.uint8) * 255)

# Lecture de l'image blanche créée pour l'utiliser comme fond
image = cv2.imread('white_img.jpg')

# Lecture des indices de fragments à ignorer depuis le fichier 'fragments_s.txt'
with open('fragments_s.txt', 'r') as fichier:
    indices_ignores = {int(ligne.strip()) for ligne in fichier.readlines()}

# Lecture des données de fragments depuis le fichier 'fragments.txt'
with open('fragments.txt', 'r') as fichier:
    lignes = fichier.readlines()

# Traitement de chaque ligne pour insérer les fragments dans l'image
for ligne in lignes:
    elements = ligne.split()  # Découpage de la ligne en éléments

    # Récupération des informations du fragment : indice, position et angle
    indice = int(elements[0])
    x = int(elements[1])
    y = int(elements[2])
    angle = float(elements[3])

    # Ignorer les fragments dont l'indice est dans la liste des indices à ignorer
    if indice in indices_ignores:
        continue
    
    # Lecture du fragment à partir du fichier image correspondant
    fragment = cv2.imread(f'frag_eroded/frag_eroded_{int(indice)}.png', cv2.IMREAD_UNCHANGED)

    # Insertion du fragment dans l'image à la position et avec l'angle spécifiés
    inserer_fragment(image, fragment, x, y, angle)

# Sauvegarde de l'image reconstruite avec les fragments insérés
cv2.imwrite('image_reconstruite.jpg', image)

# Affichage de l'image reconstruite
cv2.imshow('Image apres reconstruction', image)
cv2.waitKey(0)  # Attente d'une touche pour fermer la fenêtre
cv2.destroyAllWindows()  # Fermeture de toutes les fenêtres ouvertes
