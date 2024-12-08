#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

##########################################
# Exercice 1 : La transformée de Hough  #
#           pour les cercles            #
##########################################


# Q1 : Pour l’image four.png fournie, de taille 100 × 100 pixels, considérons que rmin = 1, rmax = 100,
# δr = 2. Combien de valeurs discrètes aura-t-on pour la coordonnée r des cercles ? Et si δr = 0.5 ?

# Calcul du nombre de valeurs discrètes pour `r` avec un pas de 2. 
# np.arange(1, 101, 2) génère une séquence de nombres de 1 à 100, inclus, avec un pas de 2.
# Cela donne les valeurs discrètes possibles pour `r`, allant de 1 à 99 avec un pas de 2.
# Le nombre de valeurs discrètes pour r sera donc le nombre d'éléments dans cette séquence.
print("Nombre de valeurs discrètes pour r avec δr = 2 (pas de 2) : ", len(np.arange(1, 101, 2)))  # Résultat pour δr = 2, soit 50 valeurs discrètes

# Calcul du nombre de valeurs discrètes pour `r` avec un pas de 0.5. 
# np.arange(1, 101, 0.5) génère une séquence de nombres de 1 à 100, inclus, avec un pas de 0.5.
# Cela donne les valeurs discrètes possibles pour `r`, allant de 1 à 99.5 avec un pas de 0.5.
# Le nombre de valeurs discrètes pour r sera donc le nombre d'éléments dans cette séquence.
print("Nombre de valeurs discrètes pour r avec δr = 0.5 (pas de 0.5) : ", len(np.arange(1, 101, 0.5)))  # Résultat pour δr = 0.5, soit 200 valeurs discrètes


# Q2 : 

# Cette question demande de calculer le nombre total de cercles possibles à partir de trois paramètres : 
# `r`, `c` et `rad`. On doit multiplier le nombre de valeurs discrètes pour `r`, `c` et `rad`.
# Le nombre de valeurs discrètes pour `r` et `c` est de 100, car les deux coordonnées varient de 1 à 100 (inclus) 
# avec un pas de 1.
# Pour `rad`, les valeurs vont de 5 à `100 * np.sqrt(2)` (environ 141), soit une plage de 137 valeurs discrètes.
# En multipliant les trois valeurs obtenues, on obtient le nombre total de cercles.
# Le calcul du nombre total de cercles pour l'image de taille 100x100 pixels est donc :
print("Nombre total de cercles possibles pour une image 100x100 avec des valeurs discrètes pour r, c et rad : ",
      len(np.arange(1, 101, 1)) *
      len(np.arange(1, 101, 1)) *
      len(np.arange(5, (100 * np.sqrt(2)) + 1, 1)))  # Résultat = 1,380,000 cercles


# Q3 : Exemple de cercles

# Cette question montre des exemples de cercles avec des valeurs spécifiques pour le rayon et les coordonnées du centre.
# Exemple 1 : Cercle de rayon 1 pixel et de centre à (1, 1).
# Exemple 2 : Cercle de rayon 30 pixels et de centre à (7, 10).
# Ces valeurs sont des exemples typiques pour illustrer le processus de détection de cercles par la méthode de Hough,
# où l'on doit tester des cercles de tailles et positions diverses dans l'image.

# Q4 : 

# Cette question se concentre sur le calcul de l'index dans l'accumulateur tridimensionnel `acc` 
# pour un cercle ayant un centre en (40, 40) et un rayon de 13 pixels.
# Avec les valeurs données : 
# - Rmin = 1 et δr = 1
# - Cmin = 1 et δc = 1
# - RADmin = 5 et δrad = 1
# Les indices dans l'accumulateur correspondent à l'index de r, c, et rad pour chaque cercle.
# Par exemple, avec un cercle de centre (40, 40) et un rayon de 13, l'index associé dans l'accumulateur sera `acc[39][39][7]`
# (l'index est basé sur les valeurs discrètes de r, c, et rad, où les valeurs commencent à 1 et sont décalées de 1 pour chaque dimension).
#
# Note : L'indice dans l'accumulateur est calculé en prenant en compte la discrétisation des paramètres `r`, `c` et `rad`.

# Affichage de l'index pour un cercle de centre (40, 40) et de rayon 13
print("L'indice dans l'accumulateur pour un cercle de rayon 13 et centre (40, 40) est : ", "acc[39][39][7]")



##############################################
# ## Exercice 2: Implementation du Detecteur ##
##############################################


# Q1:

import math


# Fonction pour afficher une image avec matplotlib
def plot_image(cv_img, g=0, title=""):
    if cv_img is None:
        print("Error: Image not found or unable to read.")  # Vérifie si l'image existe
    else:
        if (g == 1):
            rgb_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)  # Convertir BGR en RGB pour une meilleure visualisation
            plt.imshow(rgb_img)
        else:
            plt.imshow(cv_img, cmap='gray')  # Affichage en niveaux de gris si g=0
        plt.title(title)
        plt.show()


# Exemple d'application de la fonction plot_image avec une image de test
images_dir = './images/'
four = cv.imread(f'{images_dir}/four.png')  # Charge l'image "four.png"
print(np.shape(four))  # Affiche la taille de l'image
plot_image(four)  # Affiche l'image


# Calcul des gradients dans les directions X et Y avec le filtre Sobel
gray = cv.cvtColor(four, cv.COLOR_BGR2GRAY)  # Convertir l'image en niveaux de gris
sobel_x = cv.Sobel(gray, ddepth=cv.CV_16U, dx=2, dy=0)  # Applique Sobel pour détecter les bords en X
sobel_y = cv.Sobel(gray, ddepth=cv.CV_16U, dx=0, dy=2)  # Applique Sobel pour détecter les bords en Y
plot_image(sobel_x, g=1, title='X_grad')  # Affiche le gradient en X
plot_image(sobel_y, g=1, title='Y_grad')  # Affiche le gradient en Y

# Addition des deux gradients et normalisation
sobel_x_y = sobel_x + sobel_y
sobel_x_y = np.clip(sobel_x_y, 0, 255)  # Clip les valeurs au seuil [0, 255]
plot_image(sobel_x_y, g=1, title='X_Y_grad')  # Affiche la combinaison des gradients


# Fonction pour calculer les gradients dans les directions X et Y d'ordre donné
def get_grad_x_y(img, order):
    sobel_x = cv.Sobel(img, ddepth=cv.CV_64F, dx=order, dy=0, ksize=3)  # Calcul du gradient X
    sobel_y = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=order, ksize=3)  # Calcul du gradient Y

    # Calcul de la magnitude du gradient
    sobel_x_y = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalisation et conversion en image de type uint8
    sobel_x_y = np.uint8(np.clip(sobel_x_y / sobel_x_y.max() * 255, 0, 255))
    sobel_x = np.uint8(np.clip(np.abs(sobel_x) / np.abs(sobel_x).max() * 255, 0, 255))
    sobel_y = np.uint8(np.clip(np.abs(sobel_y) / np.abs(sobel_y).max() * 255, 0, 255))

    return sobel_x_y, sobel_x, sobel_y


# Fonction pour calculer la distance entre deux points (x1, y1) et (x2, y2)
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Calcul de la distance Euclidienne
    return int(round(distance))


# Fonction pour calculer l'accumulateur dans l'espace des paramètres (r, c, rayon)
def get_accumulateur(grad_img, rmin, rmax, dr, cmin, cmax, dc, radmin, radmax, drad, int_thresh):
    img_dim_y, img_dim_x = np.shape(grad_img)  # Dimensions de l'image

    # Ajuste les limites de rmax et cmax en fonction des dimensions de l'image
    rmax = min(rmax, img_dim_y)
    cmax = min(cmax, img_dim_x)
    radmax = np.ceil(min(radmax, np.sqrt(img_dim_x ** 2 + img_dim_y ** 2))).astype(int)

    max_grad = np.max(grad_img)  # Trouver la valeur maximale du gradient
    contour_pixels = np.argwhere(grad_img > int_thresh * max_grad)  # Pixels de contours (forts gradients)

    # Dimensions de l'accumulateur
    acc_dim_r = np.ceil((rmax - rmin + 1) / dr).astype(int)
    acc_dim_c = np.ceil((cmax - cmin + 1) / dc).astype(int)
    acc_dim_rad = np.ceil((radmax - radmin + 1) / drad).astype(int)

    acc = np.zeros((acc_dim_r, acc_dim_c, acc_dim_rad))  # Accumulateur initialisé à 0

    # Remplissage de l'accumulateur
    for y, x in contour_pixels:
        for r_idx, r in enumerate(range(rmin, rmax, dr)):
            for c_idx, c in enumerate(range(cmin, cmax, dc)):
                rad = int(np.sqrt((x - c) ** 2 + (y - r) ** 2))  # Calcul du rayon
                if radmin <= rad < radmax:
                    rad_idx = (rad - radmin) // drad
                    acc[r_idx, c_idx, rad_idx] += 1  # Incrémente l'accumulateur pour ce paramètre
    return acc


# Fonction pour récupérer les cercles détectés à partir de l'accumulateur
def get_cicrles(acc, rmin, dr, cmin, dc, radmin, radmax, drad, nb_circles=np.inf, acc_min_vote=26, circ_dist_r=3,
                circ_dist_c=3, circ_dist_rad=3):
    case_r = np.ceil(circ_dist_r / dr).astype(int)
    case_c = np.ceil(circ_dist_c / dc).astype(int)
    case_rad = np.ceil(circ_dist_rad / drad).astype(int)

    acc[acc < acc_min_vote] = 0  # Filtre les cercles avec un nombre de votes trop faible
    acc /= (np.arange(radmin, radmin + acc.shape[2] * drad, drad) + 1)[None, None, :]  # Normalisation de l'accumulateur

    circles = []  # Liste des cercles détectés
    visible_frac = []  # Liste des fractions visibles des cercles (pour analyser la qualité)
    i = 0
    while i < nb_circles:
        # Trouver le maximum de l'accumulateur
        ind = np.argpartition(acc.flatten(), -1)[-1]
        r_ind, c_ind, rad_ind = np.unravel_index(ind, acc.shape)
        r = rmin + (r_ind * dr)
        c = cmin + (c_ind * dc)
        rad = radmin + (rad_ind * drad)

        if acc[r_ind][c_ind][rad_ind] <= 0:
            break  # Arrêter si l'accumulateur est trop faible
        circles.append([r, c, rad])  # Ajouter le cercle détecté
        visible_frac.append(acc[r_ind][c_ind][rad_ind])  # Ajouter la fraction visible

        # Mettre à 0 les voisins du maximum pour éviter de récupérer des cercles similaires
        acc[
        max(r_ind - case_r, 0):min(r_ind + case_r + 1, acc.shape[0]),
        max(c_ind - case_c, 0):min(c_ind + case_c + 1, acc.shape[1]),
        max(rad_ind - case_rad, 0):min(rad_ind + case_rad + 1, acc.shape[2])
        ] = 0
        i += 1
    return circles, visible_frac


# Fonction pour dessiner les cercles détectés sur l'image
def draw_circles(img, c):
    for i in c:
        img = cv.circle(img, (i[1], i[0]), i[2], color=(0, 0, 255), thickness=1)  # Dessiner le cercle
    return img


# Fonction principale pour détecter les cercles
def detect_circles(img, apply_gaussian=1, gaussian_dim=(5, 5), rmin=0, rmax=0, dr=1,
                   cmin=0, cmax=0, dc=1,
                   radmin=0, radmax=0, drad=1,
                   int_thresh=0.5, acc_min_vote=10, nb_circles=np.inf, circ_dist_c=3, circ_dist_r=3, circ_dist_rad=2):
    orig_img = np.copy(img)  # Conserver une copie de l'image originale
    if apply_gaussian:
        img = cv.GaussianBlur(img, gaussian_dim, sigmaX=2, sigmaY=2)  # Appliquer un flou gaussien pour réduire le bruit

    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # Convertir en niveaux de gris pour simplifier l'analyse

    img_dim_y, img_dim_x = np.shape(img)  # Récupérer les dimensions de l'image
    rmax = min(rmax, img_dim_y)  # Ajuste rmax en fonction de la hauteur de l'image
    cmax = min(cmax, img_dim_x)  # Ajuste cmax en fonction de la largeur de l'image
    radmax = np.ceil(min(radmax, np.sqrt(img_dim_x ** 2 + img_dim_y ** 2))).astype(int)  # Maximum rayon basé sur la diagonale de l'image

    grad_img = get_grad_x_y(img, order=1)[0]  # Calcul des gradients dans les deux directions

    acc = get_accumulateur(grad_img=grad_img, rmin=rmin, rmax=rmax, dr=dr, cmin=cmin, cmax=cmax, dc=dc,
                           radmin=radmin, radmax=radmax, drad=drad, int_thresh=int_thresh)  # Calcul de l'accumulateur

    circles, visible_frac = get_cicrles(acc=acc, rmin=rmin, dr=dr, cmin=cmin, dc=dc, radmin=radmin, radmax=radmax,
                                        drad=drad, acc_min_vote=acc_min_vote, circ_dist_c=circ_dist_c,
                                        circ_dist_r=circ_dist_r, circ_dist_rad=circ_dist_rad, nb_circles=nb_circles)  # Extraction des cercles

    final_result = draw_circles(orig_img, circles)  # Dessiner les cercles détectés sur l'image originale
    return circles, visible_frac, final_result


# Tester la détection sur une image de test



# Détection des cercles sans flou gaussien, seuil de 0.5 sur "four.png"
four = cv.imread('images/four.png')
plot_image(four, g=1, title='Exercice 2 - Image originale - four.png')
print("Affichage de l'image : Exercice 2 - Image originale - four.png")

# Détection des cercles sans flou gaussien, seuil de 0.5
c, visi, res = detect_circles(four, apply_gaussian=False, gaussian_dim=(3, 3), rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10, nb_circles=4)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles sans flou gaussien, seuil 0.5')
print("Affichage de l'image : Exercice 2 - Détection des cercles sans flou gaussien, seuil 0.5")

# Détection des cercles sans flou gaussien, seuil de 0.5 sur "fourn.png"
fourn = 'images/fourn.png'
fourn = cv.imread(fourn)
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=fourn, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=4)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles sans flou gaussien sur fourn.png')
print("Affichage de l'image : Exercice 2 - Détection des cercles sans flou gaussien sur fourn.png")

# Détection des cercles avec flou gaussien, seuil de 0.5 sur "fourn.png"
fourn = 'images/fourn.png'
fourn = cv.imread(fourn)
c, visi, res = detect_circles(apply_gaussian=True, gaussian_dim=(3, 3), img=fourn, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=4)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.5')
print("Affichage de l'image : Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.5")

# Détection des cercles avec flou gaussien, seuil de 0.3 sur "fourn.png"
fourn = 'images/fourn.png'
fourn = cv.imread(fourn)
c, visi, res = detect_circles(apply_gaussian=True, gaussian_dim=(3, 3), img=fourn, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.3, acc_min_vote=10,
                              nb_circles=4)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.3')
print("Affichage de l'image : Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.3")

# Détection des cercles sans flou gaussien sur "coins.png"
coins = 'images/coins.png'
coins = cv.imread(coins)
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=coins, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=2)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles sans flou gaussien sur coins.png')
print("Affichage de l'image : Exercice 2 - Détection des cercles sans flou gaussien sur coins.png")

# Détection des cercles sans flou gaussien sur "MoonCoin.png"
MoonCoin = 'images/MoonCoin.png'
MoonCoin = cv.imread(MoonCoin)
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=MoonCoin, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=5)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles sans flou gaussien sur MoonCoin.png')
print("Affichage de l'image : Exercice 2 - Détection des cercles sans flou gaussien sur MoonCoin.png")

# Détection des cercles avec flou gaussien, seuil de 0.5 sur "MoonCoin.png"
MoonCoin = 'images/MoonCoin.png'
MoonCoin = cv.imread(MoonCoin)
c, visi, res = detect_circles(apply_gaussian=True, gaussian_dim=(3, 3), img=MoonCoin, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=5)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.5 sur MoonCoin.png')
print("Affichage de l'image : Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.5 sur MoonCoin.png")

# Détection des cercles avec flou gaussien, seuil de 0.7 sur "MoonCoin.png"
MoonCoin = 'images/MoonCoin.png'
MoonCoin = cv.imread(MoonCoin)
c, visi, res = detect_circles(apply_gaussian=True, gaussian_dim=(3, 3), img=MoonCoin, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.7, acc_min_vote=10,
                              nb_circles=5)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.7 sur MoonCoin.png')
print("Affichage de l'image : Exercice 2 - Détection des cercles avec flou gaussien, seuil 0.7 sur MoonCoin.png")

# Détection des cercles sans flou gaussien sur "MoonCoin.png" avec un rayon plus grand
MoonCoin = 'images/MoonCoin.png'
MoonCoin = cv.imread(MoonCoin)
c, visi, res = detect_circles(apply_gaussian=False, gaussian_dim=(1, 1), img=MoonCoin,
                              rmin=0, rmax=200, dr=1, cmin=0, cmax=200, dc=1, radmin=6,
                              radmax=500, drad=1, int_thresh=0.5, acc_min_vote=5, nb_circles=5,
                              circ_dist_c=2, circ_dist_r=2, circ_dist_rad=2)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles sans flou gaussien avec un rayon plus grand sur MoonCoin')
print("Affichage de l'image : Exercice 2 - Détection des cercles sans flou gaussien avec un rayon plus grand sur MoonCoin")

# Détection des cercles sans flou gaussien sur "coins2.jpg"
coins2 = 'images/coins2.jpg'
coins2 = cv.imread(coins2)
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=coins2, rmin=0, rmax=1000, dr=2, cmin=0,
                              cmax=1000, dc=2, radmin=20, radmax=500, drad=1, int_thresh=0.7, acc_min_vote=10,
                              nb_circles=6)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles sans flou gaussien sur coins2.jpg')
print("Affichage de l'image : Exercice 2 - Détection des cercles sans flou gaussien sur coins2.jpg")

# Détection des cercles avec flou gaussien sur "coins2.jpg"
coins2 = 'images/coins2.jpg'
coins2 = cv.imread(coins2)
print(np.shape(coins2))
c, visi, res = detect_circles(apply_gaussian=1, gaussian_dim=(3, 3), img=coins2, rmin=0, rmax=1000, dr=2, cmin=0,
                              cmax=1000, dc=2, radmin=20, radmax=500, drad=1, int_thresh=0.7, acc_min_vote=10,
                              nb_circles=6)
plot_image(res, g=1, title='Exercice 2 - Détection des cercles avec flou gaussien sur coins2.jpg')
print("Affichage de l'image : Exercice 2 - Détection des cercles avec flou gaussien sur coins2.jpg")

##########################################
# Exercice 3 : Temps de calcul          #
##########################################

#
# #### Question 1 :
#
# ##### **Calculs de complexité temporelle:**
#
# ##### 1) L’image contient N×N pixels. Chaque pixel est traité, ce qui donne une complexité de 𝑁².
#
# ##### 2)
# ##### * On parcourt les rayons compris entre radmin et radmax, où 0 <= radmin <= radmax <= N*N pixels.
#
# ##### OU
#
# ##### * On parcourt les pixels de contour et on calcule le rayon correspondant (notre approche dans `get_accumulateur`). Le nombre de pixels de contour est majoré par le nombre
# #####   total des pixels dans l'image N*N.
#
# ##### => Si la complexité temporelle est de N⁴ pour une image de 100px. La complexité pour une image de 600px est de 6⁴ * N⁴ = 1296 * N⁴. La complexité d'une image 6 fois plus grande qu'une image de 100 pixels est plus de 1000 fois celle d'origine.
#
# ##### Dans la suite, on va mesurer le temps d'exécution de la fonction `detect_circles` pour chaque image en utilisant les configurations ayant donné les meilleurs résultats.




# Détection des cercles sur l'image "four.png"
four = 'images/four.png'
four_img = cv.imread(four)
print(np.shape(four_img))
four_e1 = cv.getTickCount()
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=four_img, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=4)
four_e2 = cv.getTickCount()
four_time = (four_e2 - four_e1) / cv.getTickFrequency()
print(f'Execution time for four.png: {four_time}')
plot_image(res, g=1, title='Exercice 3 - Détection des cercles sur four.png')

# Détection des cercles sur l'image "fourn.png"
fourn = 'images/fourn.png'
fourn_img = cv.imread(fourn)
print(np.shape(fourn_img))
fourn_e1 = cv.getTickCount()
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=fourn_img, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=4)
fourn_e2 = cv.getTickCount()
fourn_time = (fourn_e2 - fourn_e1) / cv.getTickFrequency()
print(f'Execution time for fourn.png: {fourn_time}')
plot_image(res, g=1, title='Exercice 3 - Détection des cercles sur fourn.png')

# Détection des cercles sur l'image "MoonCoin.png"
MoonCoin = 'images/MoonCoin.png'
MoonCoin_img = cv.imread(MoonCoin)
print(np.shape(MoonCoin_img))
MoonCoin_e1 = cv.getTickCount()
c, visi, res = detect_circles(apply_gaussian=False, gaussian_dim=(1, 1), img=MoonCoin_img,
                               rmin=0, rmax=200, dr=1, cmin=0, cmax=200, dc=1, radmin=6,
                               radmax=500, drad=1, int_thresh=0.5, acc_min_vote=5, nb_circles=5,
                               circ_dist_c=2, circ_dist_r=2, circ_dist_rad=2)
MoonCoin_e2 = cv.getTickCount()
MoonCoin_time = (MoonCoin_e2 - MoonCoin_e1) / cv.getTickFrequency()
print(f'Execution time for MoonCoin.png: {MoonCoin_time}')
plot_image(res, g=1, title='Exercice 3 - Détection des cercles sur MoonCoin.png')

# Détection des cercles sur l'image "coins.png"
coins = 'images/coins.png'
coins_img = cv.imread(coins)
print(np.shape(coins_img))
coins_e1 = cv.getTickCount()
c, visi, res = detect_circles(apply_gaussian=0, gaussian_dim=(3, 3), img=coins_img, rmin=0, rmax=200, dr=1, cmin=0,
                              cmax=200, dc=1, radmin=5, radmax=500, drad=1, int_thresh=0.5, acc_min_vote=10,
                              nb_circles=2)
coins_e2 = cv.getTickCount()
coins_time = (coins_e2 - coins_e1) / cv.getTickFrequency()
print(f'Execution time for coins.png: {coins_time}')
plot_image(res, g=1, title='Exercice 3 - Détection des cercles sur coins.png')

# Détection des cercles avec flou gaussien sur "coins2.jpg"
coins2 = 'images/coins2.jpg'
coins2_img = cv.imread(coins2)
print(np.shape(coins2_img))
coins2_e1 = cv.getTickCount()
c, visi, res = detect_circles(apply_gaussian=True, gaussian_dim=(9, 9), img=coins2_img, rmin=0, rmax=1000, dr=3,
                              cmin=0, cmax=1000, dc=3, radmin=50, radmax=250, drad=3, int_thresh=0.5, acc_min_vote=8,
                              nb_circles=8, circ_dist_r=40, circ_dist_c=40, circ_dist_rad=30)
coins2_e2 = cv.getTickCount()
coins2_time = (coins2_e2 - coins2_e1) / cv.getTickFrequency()
print(f'Execution time for coins2.jpg: {coins2_time}')
plot_image(res, g=1, title='Exercice 3 - Détection des cercles avec flou gaussien sur coins2.jpg')


# Implémentation utilisant la méthode "direction du gradient"
def get_grad_x_y_ex3(img, order):
    sobel_x = cv.Sobel(img, ddepth=cv.CV_64F, dx=order, dy=0, ksize=3)
    sobel_y = cv.Sobel(img, ddepth=cv.CV_64F, dx=0, dy=order, ksize=3)

    # Calcul de la magnitude du gradient (combinaison des gradients en x et y)
    sobel_x_y = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Calcul de la direction du gradient (en radians), utilisée pour détecter les orientations des bords
    grad_dir = np.arctan2(sobel_y, sobel_x)  # Utilisation des gradients non normalisés pour des angles plus précis

    return sobel_x_y, sobel_x, sobel_y, grad_dir

def get_accumulateur_ex3(grad_img, rmin, rmax, dr, cmin, cmax,
                         dc, radmin, radmax, drad, int_thresh,
                         grad_dir, beta, angle_step_size=1):
    img_dim_y, img_dim_x = grad_img.shape
    max_grad = np.max(grad_img)
    contour_pixels = np.argwhere(grad_img > int_thresh * max_grad)  # Trouver les pixels où le gradient dépasse un seuil

    # Initialisation de l'accumulateur : une matrice pour chaque rayon, chaque position et chaque angle
    acc_dim_r = np.ceil((rmax - rmin + 1) / dr).astype(int)
    acc_dim_c = np.ceil((cmax - cmin + 1) / dc).astype(int)
    acc_dim_rad = np.ceil((radmax - radmin + 1) / drad).astype(int)
    acc = np.zeros((acc_dim_r, acc_dim_c, acc_dim_rad))

    for y, x in contour_pixels:
        grad_angle = grad_dir[y, x] + np.pi  # Ajustement de l'angle du gradient

        # Définition des limites angulaires autour de l'angle du gradient
        angle_min = grad_angle - beta
        angle_max = grad_angle + beta

        # Parcours des rayons possibles
        for r in range(radmin, radmax, drad):
            for angle in np.arange(angle_min, angle_max, angle_step_size):  # On ajuste la taille du pas pour l'angle
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                cx = int(x + r * cos_a)  # Calcul de la position en x du centre du cercle
                cy = int(y + r * sin_a)  # Calcul de la position en y du centre du cercle
                
                # Vérifier que le centre du cercle est dans les limites de l'image
                if cmin <= cx < cmax and rmin <= cy < rmax:
                    r_idx = (cy - rmin) // dr
                    c_idx = (cx - cmin) // dc
                    rad_idx = (r - radmin) // drad

                    # Mise à jour de l'accumulateur en fonction de la position du centre du cercle
                    acc[r_idx, c_idx, rad_idx] += 1

    return acc

def detect_circles_ex3(img, beta, angle_step_size=1, apply_gaussian=1, gaussian_dim=(5, 5), rmin=0, rmax=0, dr=1,
                       cmin=0, cmax=0, dc=1, radmin=0, radmax=0, drad=1,
                       int_thresh=0.5, acc_min_vote=10, nb_circles=np.inf, circ_dist_c=3,
                       circ_dist_r=3, circ_dist_rad=2):
    orig_img = np.copy(img)
    
    # Appliquer un flou gaussien si nécessaire pour réduire le bruit avant la détection
    if apply_gaussian:
        img = cv.GaussianBlur(img, gaussian_dim, sigmaX=2, sigmaY=2)

    # Conversion de l'image en niveaux de gris pour simplifier les calculs
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_dim_y, img_dim_x = np.shape(img)
    rmax = min(rmax, img_dim_y)  # Limiter rmax en fonction du nombre de lignes de l'image
    cmax = min(cmax, img_dim_x)  # Limiter cmax en fonction du nombre de colonnes de l'image
    radmax = np.ceil(min(radmax, np.sqrt(img_dim_x ** 2 + img_dim_y ** 2))).astype(
        int)  # Limiter radmax à la diagonale de l'image (distance maximale possible pour un rayon)

    # Calcul de la magnitude du gradient pour détecter les contours
    grad_img, _, __, grad_dir = get_grad_x_y_ex3(img, order=1)  # Calcul du gradient après lissage

    # Obtenir l'accumulateur qui stocke les informations sur les cercles
    acc = get_accumulateur_ex3(
        grad_img=grad_img,
        rmin=rmin,
        rmax=rmax,
        dr=dr,
        cmin=cmin,
        cmax=cmax,
        dc=dc,
        radmin=radmin,
        radmax=radmax,
        drad=drad,
        int_thresh=int_thresh,
        grad_dir=grad_dir,
        beta=beta,
        angle_step_size=angle_step_size
    )

    # Identification des cercles en fonction de la valeur de l'accumulateur
    circles = []
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            for k in range(acc.shape[2]):
                if acc[i, j, k] > acc_min_vote:  # Seuil minimum pour considérer un cercle comme détecté
                    circles.append((i, j, k))

    return circles, acc, orig_img

def plot_image(result, g=0):
    """
    Fonction pour afficher une image avec les cercles détectés.
    Le paramètre g permet d'afficher ou non l'image après le traitement.
    """
    plt.imshow(result)
    if g:
        plt.show()



