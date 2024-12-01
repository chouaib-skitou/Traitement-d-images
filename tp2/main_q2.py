import cv2
import numpy as np
import math

def main():
    #Exercice 1
    #1. Pour limage four.png fournie, de taille 100 *100 pixels, considerons que rmin = 1, rmax = 100,r =2. 
    # Combien de valeurs discretes aura-t-on pour la coordonnee r des cercles? Et si r = 05?
    rmin = 1
    rmax = 100
    rd = 2
    val = ((rmax - rmin) // rd) + 1
    print(val)
    print(f"Nombre de valeurs discrètes pour r avec un pas de {rd}: {val}")
    # Avec rd = 0.5
    rd2 = 0.5
    val2 = (rmax - rmin) // rd2 + 1
    print(f"Nombre de valeurs discrètes pour r avec un pas de {rd2}: {val2}")

    #2.Pour la meme images, en supposant que rmin = 1, rmax = 100, r = 1, cmin = 1, cmax = 100, c = 1,
    # radmin = 5 radmax = 100 2 rad = 1, quel est le nombre total de cercles quon peut decrire avec ces trois variables?
    rmin = 1
    rmax = 100
    rd = 1
    cmin = 1
    cmax = 100
    cd = 1
    radmin = 5
    radmax = 100*math.sqrt(2)
    rad = 1
    # Calcul des valeurs discrètes pour c et rad
    v_val =((rmax - rmin) // rd) + 1
    c_values = (cmax - cmin) // cd + 1
    rad_values = (radmax - radmin) // rad + 1
    # Nombre total de cercles
    print(f"Nombre total de cercles : {v_val * c_values * rad_values}")

    #3.Le tableau tridimensionnel acc associe a la case acc(i j k) le cercle situe a la i-eme valeur discrete
    # de r, la j-eme valeur discrete de c, et la k-eme valeur discrete de rad. Quel est le cercle associe au# acc(1,1,1)? Au acc(10,7,30)?
    # acc(1,1,1)
    # Pour i=1, rmin+(i-1)*dr = 1+(1-1)*1 = 1
    # Pour j=1, on a donc c=1
    # Pour k=1, on a rad =5
    # Le cercle associé est (1,1,5)

    # acc(10,7,30)
    # Pour i=10, r=10
    # Pour j=7, c=7
    # Pour k=30, rad = 34
    # Le cercle associé est (10,7,34)


    #4.Inversement, quelle est la case de laccumulateur associee au cercle centre dans le pixel (40,40) et de 
    # rayon rad = 13? Attention : les indices i,j,k doivent etre entiers.
    # on sait que r=40, c=40, rad =13#rmin = 1, cmin=1, radmin=5#deltaR = deltaC = deltaRAD = 1
    print(f"La case associée au cercle centre en (40,40) et de rayon 13: {((40-1)/1),((40-1)/1),((13-5)/1)}")

    t1 = cv2.getTickCount()
    N = 5  #cercles à affihcer à ajuster en fonction de l'image choisie dans le path
    image_path = "images/four.png" 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 1.5)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    ratio = 0.27  # Seuillage à 25% de la valeur maximale de magnitude
    threshold = ratio * magnitude.max()
    contour_pixels = (magnitude > threshold).astype(np.uint8)
    #contour_image = cv2.cvtColor(contour_pixels * 255, cv2.COLOR_GRAY2BGR)
    #cv2.imshow("Contours", contour_image)
    #cv2.waitKey(0)


    # l'accumulateur
    rmin, rmax, delta_r = 1, 100, 2  # Plage des lignes
    cmin, cmax, delta_c = 1, 100, 2  # Plage des colonnes
    radmin, radmax, delta_rad = 2, 27, 2  # rayon min et max
    # remplissage avec des zéros
    acc_shape = (
        (rmax - rmin) // delta_r + 1,
        (cmax - cmin) // delta_c + 1,
        (radmax - radmin) // delta_rad + 1,
    )
    accumulator = np.zeros(acc_shape, dtype=int)

    # Vote et remplissage de l'accumulateur
    contour_indices = np.argwhere(contour_pixels)
    for y, x in contour_indices:
        for r_idx, r in enumerate(range(rmin, rmax + 1, delta_r)):
            for c_idx, c in enumerate(range(cmin, cmax + 1, delta_c)):
                rad = int(np.sqrt((x - c) ** 2 + (y - r) ** 2))
                if radmin <= rad <= radmax:
                    rad_idx = (rad - radmin) // delta_rad
                    accumulator[r_idx, c_idx, rad_idx] += magnitude[y, x] / rad  # Pondération pour les cercles de grandes tailles

    # maxima locaux dans l'accumulateur
    def is_local_maximum(i, j, k):
        return accumulator[i, j, k] == np.max(accumulator[i-1:i+2, j-1:j+2, k-1:k+2])

    local_maxima = np.zeros(accumulator.shape, dtype=bool)
    for i in range(1, accumulator.shape[0] - 1):
        for j in range(1, accumulator.shape[1] - 1):
            for k in range(1, accumulator.shape[2] - 1):
                if accumulator[i, j, k] > 0 and is_local_maximum(i, j, k):
                    local_maxima[i, j, k] = True

    #Sélection des N cercles
    maxima_indices = np.argwhere(local_maxima)
    maxima_values = accumulator[local_maxima]
    sorted_maxima_indices = np.argsort(maxima_values)[::-1][:N]  # Tri
    top_circles = maxima_indices[sorted_maxima_indices]

    # Affichage des cercles détectés
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for r_idx, c_idx, rad_idx in top_circles:
        r = rmin + r_idx * delta_r
        c = cmin + c_idx * delta_c
        rad = radmin + rad_idx * delta_rad
        center = (c, r)
        cv2.circle(output_image, center, rad, (0, 0, 255), 2)  # cercle rouge
        cv2.circle(output_image, center, 2, (0, 255, 0), 2)    # centre vert

    t2 = cv2.getTickCount()
    t_tot = (t2 - t1) / cv2.getTickFrequency()
    print(f"Temps de calcul : {t_tot:.2f} secondes")

    cv2.imshow("Cercles detectés", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
