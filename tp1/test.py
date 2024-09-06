import cv2
import numpy as np

# Étape 1 : Créer une image vide (ex: 800x600, avec 3 canaux pour RGB)
image_vide = np.zeros((600, 800, 3), dtype=np.uint8)

# Optionnel : remplir l'image vide avec une couleur (ex: blanc)
image_vide[:] = (255, 255, 255)  # Blanc

# Étape 2 : Charger les petites images (assurez-vous que les chemins sont corrects)
petite_image_1 = cv2.imread('frag_eroded/frag_eroded_13.png')
petite_image_2 = cv2.imread('frag_eroded/frag_eroded_30.png')

# Étape 3 : Redimensionner les petites images si nécessaire
# petite_image_1 = cv2.resize(petite_image_1, (100, 100))  # Redimensionner à 100x100 pixels
# petite_image_2 = cv2.resize(petite_image_2, (150, 150))  # Redimensionner à 150x150 pixels

# Étape 4 : Superposer les petites images à des positions spécifiques sur l'image vide
# Exemple : Superposer la première petite image à (50, 50)
image_vide[50:225, 50:225] = petite_image_1

# Superposer la deuxième petite image à (300, 200)
image_vide[200:353, 300:453] = petite_image_2

# Afficher l'image finale
cv2.imshow("Image finale", image_vide)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Enregistrer l'image finale si nécessaire
cv2.imwrite('image_superposee.png', image_vide)
