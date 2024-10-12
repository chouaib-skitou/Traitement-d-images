def read_fragments(file):
    fragments = []
    with open(file, 'r') as f:
        for line in f:
            index, x, y, angle = map(float, line.strip().split())
            fragments.append((x, y, angle))
    return fragments

def fragment_well_placed(solution, reference, tol_x, tol_y, tol_angle):
    return (abs(solution[0] - reference[0]) <= tol_x and
            abs(solution[1] - reference[1]) <= tol_y and
            abs(solution[2] - reference[2]) <= tol_angle)

def calculate_accuracy(solution_file, reference_file, tol_x=1, tol_y=1, tol_angle=1):
    solution_fragments = read_fragments(solution_file)
    reference_fragments = read_fragments(reference_file)
    
    well_placed_fragments = 0
    total_fragments = len(reference_fragments)
    
    for solution in solution_fragments:
        for reference in reference_fragments:
            if fragment_well_placed(solution, reference, tol_x, tol_y, tol_angle):
                well_placed_fragments += 1
                break

    if well_placed_fragments == total_fragments:
        print("The solution is correct")
    else:
        print("The solution is not correct")

calculate_accuracy('solution.txt', 'fragments.txt')
