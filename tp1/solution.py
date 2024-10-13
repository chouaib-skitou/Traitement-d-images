import math

# Lecture du fichier de fragments
def read_fragments(file):
    fragments = []
    with open(file, 'r') as f:
        for line in f:
            index, x, y, angle = map(float, line.strip().split())
            fragments.append((x, y, angle))
    return fragments

# Vérifie si un fragment est bien positionné selon les tolérances
def fragment_well_placed(solution, reference, tol_x, tol_y, tol_angle):
    return (abs(solution[0] - reference[0]) <= tol_x and
            abs(solution[1] - reference[1]) <= tol_y and
            abs(solution[2] - reference[2]) <= tol_angle)

# Calcul de la précision
def calculate_accuracy(solution_file, reference_file, tol_x=1, tol_y=1, tol_angle=1):
    solution_fragments = read_fragments(solution_file)
    reference_fragments = read_fragments(reference_file)
    
    well_placed_area = 0
    total_area = len(reference_fragments)  # Considérons l'aire comme le nombre de fragments
    incorrect_area = 0
    
    # Calcul des fragments bien positionnés
    for solution in solution_fragments:
        fragment_correct = False
        for reference in reference_fragments:
            if fragment_well_placed(solution, reference, tol_x, tol_y, tol_angle):
                well_placed_area += 1
                fragment_correct = True
                break
        if not fragment_correct:
            incorrect_area += 1  # Fragments qui ne sont pas dans la fresque
    
    # Calcul de la précision p
    p = well_placed_area / total_area
    
    # Affichage des résultats
    print(f"Tolérances utilisées : Δx = {tol_x}, Δy = {tol_y}, Δα = {tol_angle}")
    print(f"Surface des fragments bien localisés : {well_placed_area}")
    print(f"Surface totale des fragments : {total_area}")
    print(f"Surface des fragments mal placés : {incorrect_area}")
    print(f"Précision p : {p}")
    
    return p

# Exécution avec des fichiers exemples
calculate_accuracy('solution.txt', 'fragments.txt')
