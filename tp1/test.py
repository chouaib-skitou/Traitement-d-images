import cv2
import numpy as np
import os

def analyser_fichier(fichier):
    # Ouvrir le fichier et lire son contenu
    with open(fichier, 'r') as f:
        lignes = f.readlines()

    # Initialiser une liste pour stocker les informations des fragments
    fragments = []

    # Parcourir chaque ligne du fichier
    for ligne in lignes:
        # Diviser la ligne en utilisant les espaces comme séparateurs
        elements = ligne.split()
        
        # Vérifier si la ligne a bien 4 éléments (index, posx, posy, angle)
        if len(elements) == 4:
            # Extraire l'index, posx, posy, et angle
            index = int(elements[0])
            posx = int(elements[1])
            posy = int(elements[2])
            angle = float(elements[3])
            
            # Ajouter les informations à la liste des fragments
            fragments.append((index, posx, posy, angle))

    return fragments

def coller_images(image_vide, fragments, dossier_fragments):
    for fragment in fragments:
        index, posx, posy, angle = fragment

        # Charger l'image associée à l'index
        chemin_image = os.path.join(dossier_fragments, f'frag_eroded_{index}.png')
        petite_image = cv2.imread(chemin_image, cv2.IMREAD_UNCHANGED)

        # Vérifier si l'image a 4 canaux (RGBA)
        if petite_image is not None and petite_image.shape[2] == 4:
            # Extraire les canaux couleur et alpha
            petite_image_rgb = petite_image[:, :, :3]  # Couleur (RGB)
            alpha_mask = petite_image[:, :, 3]  # Canal alpha
            
            # Rotation de l'image
            (h, w) = petite_image_rgb.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            petite_image_rgb_rotated = cv2.warpAffine(petite_image_rgb, M, (w, h))
            alpha_mask_rotated = cv2.warpAffine(alpha_mask, M, (w, h))

            # Calcul de l'espace disponible à partir de (posx, posy)
            espace_disponible_vertical = image_vide.shape[0] - posy
            espace_disponible_horizontal = image_vide.shape[1] - posx
            
            # Si la petite image dépasse les bords, on découpe la partie qui rentre
            hauteur_petite_image, largeur_petite_image = petite_image_rgb_rotated.shape[:2]
            hauteur_finale = min(hauteur_petite_image, espace_disponible_vertical)
            largeur_finale = min(largeur_petite_image, espace_disponible_horizontal)
            
            petite_image_rgb_decoupee = petite_image_rgb_rotated[0:hauteur_finale, 0:largeur_finale]
            alpha_mask_decoupee = alpha_mask_rotated[0:hauteur_finale, 0:largeur_finale]

            # Créer un masque inversé pour l'image de fond
            masque_inv = cv2.bitwise_not(alpha_mask_decoupee)

            # Extraire la région de l'image vide où l'on veut coller l'image
            region_arriere_plan = image_vide[posy:posy + hauteur_finale, posx:posx + largeur_finale]

            # Appliquer le masque sur la région d'arrière-plan
            arriere_plan = cv2.bitwise_and(region_arriere_plan, region_arriere_plan, mask=masque_inv)

            # Appliquer le masque alpha sur la petite image
            avant_plan = cv2.bitwise_and(petite_image_rgb_decoupee, petite_image_rgb_decoupee, mask=alpha_mask_decoupee)

            # Combiner les deux images
            image_superposee = cv2.add(arriere_plan, avant_plan)

            # Placer l'image combinée dans l'image vide
            image_vide[posy:posy + hauteur_finale, posx:posx + largeur_finale] = image_superposee

        else:
            print(f"Erreur : Impossible de charger l'image ou l'image ne contient pas de canal alpha pour l'index {index}.")

# Étape 1 : Créer une image vide (775x1707, avec 3 canaux pour RGB)
image_vide = np.zeros((3000, 4500, 3), dtype=np.uint8)
image_vide[:] = (255, 255, 255)  # Blanc

# Étape 2 : Lire les informations des fragments depuis le fichier
fragments = analyser_fichier("fragments.txt")

# Étape 3 : Coller les images sur l'image vide
dossier_fragments = "frag_eroded"
coller_images(image_vide, fragments, dossier_fragments)

# Afficher l'image finale
cv2.imshow("Image finale", image_vide)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Enregistrer l'image finale si nécessaire
cv2.imwrite('image_superposee.png', image_vide)
