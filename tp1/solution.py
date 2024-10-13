import math

# Lecture du fichier de fragments
def read_fragments(file):
    fragments = []
    with open(file, 'r') as f:
        for line in f:
            index, x, y, angle = map(float, line.strip().split())
            fragments.append((int(index), x, y, angle))  # Inclure l'index du fragment
    return fragments

# Lecture des indices des fragments à ignorer
def read_ignored_fragments(file):
    with open(file, 'r') as f:
        return {int(line.strip()) for line in f.readlines()}

# Vérifie si un fragment est bien positionné selon les tolérances
def fragment_well_placed(solution, reference, tol_x, tol_y, tol_angle):
    return (abs(solution[1] - reference[1]) <= tol_x and  # x
            abs(solution[2] - reference[2]) <= tol_y and  # y
            abs(solution[3] - reference[3]) <= tol_angle)  # angle

# Calcul de la précision
def calculate_accuracy(solution_file, reference_file, ignored_fragments_file, tol_x=1, tol_y=1, tol_angle=1):
    solution_fragments = read_fragments(solution_file)
    reference_fragments = read_fragments(reference_file)
    ignored_fragments = read_ignored_fragments(ignored_fragments_file)

    well_placed_area = 0
    total_area = len(reference_fragments)  # Surface totale des fragments de la fresque
    incorrect_area = 0
    ignored_area = 0

    # Calcul des fragments bien positionnés
    for solution in solution_fragments:
        index = solution[0]

        # Si le fragment fait partie de ceux ignorés
        if index in ignored_fragments:
            ignored_area =+ 1
            continue
        
        # Vérification de la localisation correcte
        fragment_correct = False
        for reference in reference_fragments:
            if fragment_well_placed(solution, reference, tol_x, tol_y, tol_angle):
                well_placed_area += 1
                fragment_correct = True
                break
        
        # Si le fragment n'est pas bien placé et n'est pas ignoré, on l'ajoute à incorrect_area
        if not fragment_correct:
            incorrect_area += 1

    # Calcul de la précision p = (surface fragments bien localises - surface fragments positionnes mais nappartenant pas a la fresque) / surface tous les fragments de la fresque
    p = (well_placed_area - ignored_area) / total_area

    # Affichage des résultats
    print(f"Tolérances utilisées : Δx = {tol_x}, Δy = {tol_y}, Δα = {tol_angle}")
    print(f"Surface des fragments bien localisés : {well_placed_area}")
    print(f"Surface totale des fragments de la fresque : {total_area}")
    print(f"Surface des fragments mal placés : {incorrect_area}")
    print(f"Précision p (plus elle est proche de 1 plus la qualité est élevée ) : {p}") 
    
    return p

# Exécution avec des fichiers exemples
calculate_accuracy('solution.txt', 'fragments.txt', 'fragments_s.txt')
