import numpy as np


rows = 5
columns = 4

coverage_matrix = np.zeros((rows, columns))


def update_coverage_matrix(row, column, value):

    # print("\n--- UPDATE COVERAGE ---")
    # print("Posizione agente:", (row, column))
    # print("Valore precedente:", coverage_matrix[row, column])

    if coverage_matrix[row, column] == 0:
        coverage_matrix[row, column] = value
        # print(">>> CELLA NUOVA: impostata a 1")
    # else:
    #     print(">>> CELLA GIA' VISITATA")

    # print("Coverage matrix:")
    # print(coverage_matrix)
    return coverage_matrix


def get_coverage_mask():

    mask = 0

    for r in range(rows):
        for c in range(columns):

            if coverage_matrix[r, c] == 1:

                index = r * columns + c

                mask |= (1 << index)

    return mask


def is_coverage_completed():
    return np.all(coverage_matrix != 0)


def percentage_coverage():
    visited_cells = np.count_nonzero(coverage_matrix != 0)
    total_cells = rows * columns

    return visited_cells / total_cells
