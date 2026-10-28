import os
import pickle
import numpy as np
import time
import math
import sys
from collections import defaultdict, deque

# import rclpy

import coverage as cov

from coverage import (
    update_coverage_matrix,
    is_coverage_completed,
    percentage_coverage
)
import random

# ============================================================
# ENVIRONMENT
# ============================================================

rows = 5
columns = 4

actions = [
    "forward",
    "turn_left",
    "turn_right",
    "acoustic",
    "visual",
    "projection"
]


def configure_training():
    global task
    global start_row, start_column
    global start_ped_row, start_ped_column

    print("Grid:", rows, "rows x", columns, "columns")
    print("Le coordinate partono da 0.")

    task = int(input("Inserisci il task: 1 (copertura), 2 (passing-side): "))

    start_row = int(input("Inserisci la riga della posizione iniziale: "))
    start_column = int(input("Inserisci la colonna della posizione iniziale: "))

    start_ped_row = int(input("Inserisci la riga della posizione illegale: "))
    start_ped_column = int(input("Inserisci la colonna della posizione illegale: "))

    if task not in (1, 2):
        raise ValueError("Il task deve essere 1 o 2")

    if not (
        0 <= start_row < rows
        and 0 <= start_column < columns
    ):
        raise ValueError("La posizione iniziale non è valida.")

    if not (
        0 <= start_ped_row < rows
        and 0 <= start_ped_column < columns
    ):
        raise ValueError("La posizione illegale non è valida.")

    if (start_row, start_column) == (start_ped_row, start_ped_column):
        raise ValueError(
            "La posizione iniziale non può coincidere "
            "con quella del pedone."
        )


# ============================================================
# Q-LEARNING TABLES
# ============================================================

# Q-values for each state-action pair
n_q_actions = len(actions)

q_values = defaultdict(
    lambda: np.zeros(n_q_actions)
)

# Values used to identify safe/unsafe actions
safe_actions = np.zeros((rows, columns, len(actions)))


def is_passing_completed(
    distance,
    robot_visibility,
    robot_orientation,
    old_robot_column,
    new_robot_column,
    old_robot_row,
    new_robot_row,
    ped_visibility
):

    cond_distance = robot_visibility and distance == 1

    cond_horizontal = (
        new_robot_column != old_robot_column
        and new_robot_row == old_robot_row
    )

    cond_vertical = (
        new_robot_column == old_robot_column
        and new_robot_row != old_robot_row
        and not ped_visibility
    )

    cond_orientation = robot_orientation in ("LEFT", "RIGHT")

    result = cond_distance and (
        cond_horizontal or cond_vertical or cond_orientation
    )

    return result


def go_forward(row_index, column_index, orientation):

    new_row = row_index
    new_column = column_index

    if orientation == "UP":
        new_row -= 1
    elif orientation == "RIGHT":
        new_column += 1
    elif orientation == "DOWN":
        new_row += 1
    elif orientation == "LEFT":
        new_column -= 1

    if not (
        0 <= new_row < rows
        and 0 <= new_column < columns
    ):
        return row_index, column_index, False

    return new_row, new_column, True


# ============================================================
# TURN LEFT
# ============================================================
def turn_left(
    robot_orientation
):
    orientations = ["UP", "LEFT", "DOWN", "RIGHT"]

    current_index = orientations.index(robot_orientation)

    new_orientation = orientations[
        (current_index + 1) % len(orientations)
    ]

    # La rotazione sul posto è sempre consentita.
    return new_orientation, True


# ============================================================
# TURN RIGHT
# ============================================================

def turn_right(
    robot_orientation
):
    orientations = ["UP", "RIGHT", "DOWN", "LEFT"]

    current_index = orientations.index(robot_orientation)

    new_orientation = orientations[
        (current_index + 1) % len(orientations)
    ]

    # La rotazione sul posto è sempre consentita.
    return new_orientation, True


def comm_effective(action, robot_vis, ped_sees_robot):
    if action == 3:
        return True                          # acoustic
    if action == 4:
        return robot_vis and ped_sees_robot  # visual
    if action == 5:
        return ped_sees_robot                # projection
    return False


# ============================================================
# REWARD FUNCTION
# ============================================================
def get_reward(
    old_robot_row,
    old_robot_column,
    new_robot_row,
    new_robot_column,
    ped_row,
    ped_column,
    previous_distance,
    distance,
    action,
    previous_action,
    previous_robot_visibility,
    robot_visibility,
    robot_orientation,
    robot_moved,
    ped_visibility,
    old_awareness,
    awareness
):
    # Inizializzazione
    reward = 0
    done = False

    # ========================================================
    # REWARD GENERICI
    # ========================================================

    # Risultati riferiti all'azione realmente eseguita
    turn_left_result = action == 1
    turn_right_result = action == 2

    # Solo forward prevede avanzamento
    go_forward_result = robot_moved if action == 0 else False

    # Robot fermo
    # if (old_robot_row == new_robot_row and old_robot_column == new_robot_column):
    #     reward = -0.1

    if action in (3, 4, 5):
        if awareness > old_awareness:
            reward -= 0.03
        else:
            reward -= 0.15

    # ========================================================
    # TASK 1: COVERAGE - REWARD SHAPING
    # ========================================================

    if task == 1:

        # Riduce il numero di step totali
        reward = -0.01

        if action == 0:

            if robot_moved:  # Avanzamento

                destination_status = cov.coverage_matrix[
                    new_robot_row, new_robot_column
                ]

                if destination_status == 0:
                    reward += 1.0   # nuova cella
                else:
                    reward -= 0.2   # overlap
            else:
                reward -= 0.5   # forward bloccato / muro

        # # Se effettua due azioni turn consecutive
        elif action != 0:  # Turn
            if previous_action is not None and previous_action != 0:
                reward = -0.1
            else:
                reward = 0.0

        done = False

    # ========================================================
    # TASK 2: PASSING-SIDE
    # ========================================================
    elif task == 2:

        if (
            robot_visibility
            and robot_moved
            and abs(new_robot_row - ped_row) == 1
            and abs(new_robot_column - ped_column) == 1
        ):
            reward = 10
            done = True

        if DEBUG:
            print(
                f"[PASSING DEBUG] "
                f"osservazione_precedente={previous_robot_visibility}, "
                f"osservazione_attuale={robot_visibility}, "
                f"rotazione_sinistra={turn_left_result}, "
                f"rotazione_destra={turn_right_result}, "
                f"avanzamento={go_forward_result}, "
                f"done={done}"
            )

    # ========================================================
    # REWARD COMUNICAZIONE
    # ========================================================

    return reward, done


def get_awareness(
    ped_visibility,
    robot_visibility,
    action,
    awareness
):
    passing_action_names = {
        3: "acoustic",
        4: "visual",
        5: "projection"
    }

    communication = passing_action_names.get(action)

    # Comunicazione acustica: sempre efficace.
    if communication == "acoustic":
        awareness = 1.0

    # Il pedone non guarda il robot:
    # l'awareness diminuisce indipendentemente dall'azione.
    elif not ped_visibility:
        awareness = max(0.0, awareness - 0.1)

    # Il pedone guarda il robot, ma il robot non comunica.
    elif communication is None:
        awareness = max(0.0, awareness - 0.1)

    # Proiezione: efficace se il pedone vede il robot.
    elif communication == "projection":
        awareness = 1.0

    # Comunicazione visiva: efficace se entrambi si vedono.
    elif communication == "visual" and robot_visibility:
        awareness = 1.0

    # Comunicazione acustica: efficase se entrambi non si vedono
    # elif communication == "acoustic" and not robot_visibility:
    #     awareness = 1.0

    # Comunicazione visiva inefficace.
    else:
        awareness = max(0.0, awareness - 0.1)

    return awareness


# ============================================================
# INITIAL POSITION
# ============================================================
def getStartingRobotLocation():

    row_index = start_row
    column_index = start_column

    return row_index, column_index


# ============================================================
# OBSTACLE POSITION
# ============================================================

def getStartingPedestrianLocation():

    row_index = start_ped_row
    column_index = start_ped_column

    return row_index, column_index


# ============================================================
# DECAY EPSILON
# ============================================================

def decay_epsilon(episode, epsilon_decay_episodes, min_epsilon=0.05):

    if episode >= epsilon_decay_episodes:
        return min_epsilon

    return 1.0 - (
        (1.0 - min_epsilon) * episode / epsilon_decay_episodes
    )


def decay_epsilon_step(
    global_step,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=200000,
):
    if global_step >= epsilon_decay:
        return epsilon_end

    progress = global_step / epsilon_decay
    return epsilon_start - (epsilon_start - epsilon_end) * progress


def get_legal_actions(robot_row, robot_column, robot_orientation):
    legal_actions = []

    for action_index in range(len(actions)):
        (
            new_robot_row,
            new_robot_column,
            new_robot_orientation,
            moved
        ) = get_next_position(
            robot_row,
            robot_column,
            robot_orientation,
            action_index
        )

        # Escludi l'avanzamento soltanto se il robot
        # non può spostarsi nella direzione corrente.
        if action_index == 0 and not moved:
            continue

        if not (0 <= new_robot_row < rows and 0 <= new_robot_column < columns):
            continue

        # Escludi gli orientamenti rivolti verso l'esterno.
        # facing_outward = (
        #     (new_robot_row == 0 and new_robot_orientation == "UP")
        #     or
        #     (new_robot_row == rows - 1 and new_robot_orientation == "DOWN")
        #     or
        #     (new_robot_column == 0 and new_robot_orientation == "LEFT")
        #     or
        #     (new_robot_column == columns - 1 and new_robot_orientation == "RIGHT")
        # )

        # if facing_outward:
        #     continue

        legal_actions.append(action_index)

    return legal_actions


# ============================================================
# ACTION SELECTION
# ============================================================

def get_action(
    state,
    epsilon,
    robot_row,
    robot_column,
    robot_orientation
):
    legal_actions = get_legal_actions(
        robot_row,
        robot_column,
        robot_orientation
    )

    if not legal_actions:
        return None

    state_key = tuple(state)

    # Esplorazione
    if np.random.random() < epsilon:
        return int(np.random.choice(legal_actions))

    # Sfruttamento
    legal_q_values = q_values[state_key][legal_actions]

    best_index = int(np.argmax(legal_q_values))

    return int(legal_actions[best_index])


# ============================================================
# STATE TRANSITION
# ============================================================
def get_next_position(
    robot_row,
    robot_column,
    robot_orientation,
    action
):
    new_robot_row = robot_row
    new_robot_column = robot_column
    new_orientation = robot_orientation
    robot_moved = False

    if action == 0:  # forward
        new_robot_row, new_robot_column, robot_moved = go_forward(
            robot_row,
            robot_column,
            robot_orientation
        )

    elif action == 1:  # turn_left
        new_orientation, _ = turn_left(robot_orientation)

    elif action == 2:  # turn_right
        new_orientation, _ = turn_right(robot_orientation)

    elif action in (3, 4, 5):  # Azioni comunicative
        pass

    else:
        raise ValueError(f"Indice azione non valido: {action}")

    return (
        new_robot_row,
        new_robot_column,
        new_orientation,
        robot_moved
    )


def discretize_distance(distance):
    if not math.isfinite(float(distance)):
        return "UNKNOWN"

    distance = float(distance)

    if distance <= 1.0:
        return "NEAR"
    elif distance <= 2.0:
        return "MEDIUM"
    else:
        return "FAR"


def discretize_awareness(awareness):

    awareness = round(float(awareness), 1)

    if awareness == 0.0:
        return "NOT AWARE"
    elif awareness == 1.0:
        return "AWARE"
    else:
        return "A LITTLE AWARE"


def is_front_cell_visited(robot_row, robot_column, robot_orientation):
    """
    Restituisce True se la cella immediatamente davanti al robot
    è già stata visitata, False altrimenti.

    Una cella è considerata visitata se coverage_matrix contiene 1.
    Le celle occupate dal pedone (-1) non sono considerate visitate
    dal robot.
    """

    front_row, front_column, robot_moved = go_forward(
        robot_row,
        robot_column,
        robot_orientation
    )

    # Davanti al robot c'è il bordo della griglia.
    if not robot_moved:
        return False

    return cov.coverage_matrix[front_row, front_column] == 1


def build_state(
    distance,
    robot_orientation,
    ped_orientation,
    robot_visibility,
    ped_visibility,
    awareness,
    front_cell_visited,
    robot_row,
    robot_column
):
    # distance_state = discretize_distance(distance)
    # awareness_state = discretize_awareness(awareness)
    distance = round(float(distance), 2)
    awareness = round(float(awareness), 1)
    # percentage = round(float(percentage_coverage()), 3)

    path, _ = bfs_grid(
        (robot_row, robot_column),
        lambda row, column: cov.coverage_matrix[row, column] == 0
    )

    if task == 1:
        rel_dir, dist_free = None, 0
        if path is not None:
            nr, nc = path[1]
            abs_dir = {(-1, 0): "UP", (0, 1): "RIGHT", (1, 0): "DOWN", (0, -1): "LEFT"}[(nr - robot_row, nc - robot_column)]
            rel_dir = (DIRS.index(abs_dir) - DIRS.index(robot_orientation)) % 4  # 0=davanti,1=dx,2=dietro,3=sx
            dist_free = min(len(path) - 1, 5)

    state = (
        distance,
        robot_orientation,
        ped_orientation,
        bool(robot_visibility),
        bool(ped_visibility),
        awareness
    )

    if task == 1:
        # state += (bool(front_cell_visited),)
        # coverage_state = tuple(
        #     cov.coverage_matrix.flatten().tolist()
        # )
        # state += (coverage_state,)
        # state += (bool(is_coverage_completed()),)
        # state += (percentage,)
        state += (rel_dir, dist_free)

    return state


# ============================================================
# SAFE ACTIONS
# ============================================================

def updates_safe_actions(
    current_row_index,
    current_column_index,
    reward,
    action_index
):

    # Illegal action
    if reward == -1:

        safe_actions[
            current_row_index,
            current_column_index,
            action_index
        ] = -1

    return safe_actions


# ============================================================
# BFS PER CAPIRE DOVE SI TROVA LA PROSSIMA CELLA NON VISITATA
# ============================================================
DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]


def bfs_grid(start, is_target):
    """
    start: (riga, colonna) del robot.
    is_target: funzione (row, column) -> bool, True per le celle da raggiungere
               (qui: coverage_matrix[row, column] == 0).
    Ritorna (percorso, parent): percorso verso la cella target più vicina
    (lista di celle, start incluso) oppure None; parent è l'albero della BFS.
    """
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        for _ in range(len(queue)):          # un livello alla volta, come nel tuo codice
            row, column = queue.popleft()

            # Aggiunge la cella non ancora visitata (= 0) come nodo del percorso
            if (row, column) != start and is_target(row, column):
                # ricostruisce il percorso risalendo i parent
                path = []
                node = (row, column)
                while node is not None:
                    path.append(node)
                    node = parent[node]
                return path[::-1], parent

            for orientation in DIRS:
                new_row, new_column, moved = go_forward(row, column, orientation)

                # Prende la prossima cella che il robot può visitare forward
                # Aggiorna le strutture visited, parent e queue
                if moved and (new_row, new_column) not in visited:
                    visited.add((new_row, new_column))
                    parent[(new_row, new_column)] = (row, column)
                    queue.append((new_row, new_column))

    return None, parent


def memorize_robot_position(
    awareness,
    robot_row,
    robot_column,
    old_robot_row=None,
    old_robot_column=None
):
    memorized = False
    random_value = bool(np.random.randint(0, 2))

    # Probabilità del 50% di memorizzare la posizione
    # se l'awareness supera 0.7.
    if awareness > 0.8 and random_value:
        old_robot_row = robot_row
        old_robot_column = robot_column
        memorized = True

    # Se non viene creata una nuova memoria, conserva quella precedente.
    elif old_robot_row is not None and old_robot_column is not None:
        memorized = True

    return memorized, old_robot_row, old_robot_column


# ============================================================
# UPDATE PEDESTRIAN POSITION
# ============================================================
def update_pedestrian_position(
    ped_row,
    ped_column,
    remembered_robot_row,
    remembered_robot_column,
    robot_row,
    robot_column,
    ped_orientation,
    awareness
):
    # Aggiorna la memoria della posizione del robot.
    memorized, remembered_robot_row, remembered_robot_column = memorize_robot_position(
        awareness,
        robot_row,
        robot_column,
        remembered_robot_row,
        remembered_robot_column
    )

    # Posizioni vietate.
    forbidden_positions = {(robot_row, robot_column)}

    if memorized:
        forbidden_positions.add((remembered_robot_row, remembered_robot_column))

    # Azioni possibili.
    actions = ("forward", "turn_left", "turn_right")

    valid_actions = []

    for action in actions:
        # Determina l'orientamento dopo l'azione.
        if action == "forward":
            candidate_orientation = ped_orientation
        elif action == "turn_left":
            candidate_orientation, _ = turn_left(ped_orientation)
        else:
            candidate_orientation, _ = turn_right(ped_orientation)

        # Calcola la cella davanti al pedone.
        candidate_row, candidate_column, candidate_moved = go_forward(
            ped_row,
            ped_column,
            candidate_orientation
        )

        candidate_position = (candidate_row, candidate_column)

        # Accetta soltanto azioni che consentono di avanzare
        # in una cella valida e non vietata.
        if (
            candidate_moved
            and candidate_position not in forbidden_positions
        ):
            valid_actions.append(
                (
                    candidate_row,
                    candidate_column,
                    candidate_orientation
                )
            )

    # Inizializza la posizione finale.
    new_ped_row = ped_row
    new_ped_column = ped_column
    new_orientation = ped_orientation
    ped_moved = False

    # Sceglie casualmente una delle azioni valide.
    if valid_actions:
        selected = valid_actions[np.random.randint(len(valid_actions))]

        new_ped_row, new_ped_column, new_orientation = selected
        ped_moved = True

    else:
        correction_actions = []

        for action in ("turn_left", "turn_right"):

            if action == "turn_left":
                candidate_orientation, _ = turn_left(ped_orientation)
            else:
                candidate_orientation, _ = turn_right(ped_orientation)

            candidate_row, candidate_column, candidate_moved = go_forward(
                ped_row,
                ped_column,
                candidate_orientation
            )

            candidate_position = (candidate_row, candidate_column)

            if (
                candidate_moved
                and candidate_position not in forbidden_positions
            ):
                correction_actions.append(candidate_orientation)

        if correction_actions:
            new_orientation = random.choice(correction_actions)

    return (
        new_ped_row,
        new_ped_column,
        new_orientation,
        ped_moved,
        remembered_robot_row,
        remembered_robot_column
    )


def calculate_distance(
    robot_row,
    robot_column,
    pedestrian_row,
    pedestrian_column,
    robot_orientation
):
    if robot_orientation == "UP":
        pedestrian_in_view = (
            pedestrian_row < robot_row and abs(pedestrian_column - robot_column) <= 1
        )

    elif robot_orientation == "DOWN":
        pedestrian_in_view = (
            pedestrian_row > robot_row and abs(pedestrian_column - robot_column) <= 1
        )

    elif robot_orientation == "LEFT":
        pedestrian_in_view = (
            pedestrian_column < robot_column and abs(pedestrian_row - robot_row) <= 1
        )

    elif robot_orientation == "RIGHT":
        pedestrian_in_view = (
            pedestrian_column > robot_column and abs(pedestrian_row - robot_row) <= 1
        )

    else:
        return None

    if not pedestrian_in_view:
        return None

    return math.sqrt(
        (pedestrian_row - robot_row) ** 2 + (pedestrian_column - robot_column) ** 2
    )


def get_robot_visibility(
    robot_row,
    robot_column,
    pedestrian_row,
    pedestrian_column,
    robot_orientation
):
    if robot_orientation == "UP":
        pedestrian_in_view = (
            pedestrian_row < robot_row and abs(pedestrian_column - robot_column) <= 1
        )

    elif robot_orientation == "DOWN":
        pedestrian_in_view = (
            pedestrian_row > robot_row and abs(pedestrian_column - robot_column) <= 1
        )

    elif robot_orientation == "LEFT":
        pedestrian_in_view = (
            pedestrian_column < robot_column and abs(pedestrian_row - robot_row) <= 1
        )

    elif robot_orientation == "RIGHT":
        pedestrian_in_view = (
            pedestrian_column > robot_column and abs(pedestrian_row - robot_row) <= 1
        )

    else:
        return None

    return pedestrian_in_view


def get_pedestrian_visibility(
    robot_row,
    robot_column,
    pedestrian_row,
    pedestrian_column,
    ped_orientation
):
    if ped_orientation == "UP":
        robot_in_view = (
            robot_row < pedestrian_row and abs(pedestrian_column - robot_column) <= 1
        )

    elif ped_orientation == "DOWN":
        robot_in_view = (
            robot_row > pedestrian_row and abs(pedestrian_column - robot_column) <= 1
        )

    elif ped_orientation == "LEFT":
        robot_in_view = (
            robot_column < pedestrian_column and abs(pedestrian_row - robot_row) <= 1
        )

    elif ped_orientation == "RIGHT":
        robot_in_view = (
            robot_column > pedestrian_column and abs(pedestrian_row - robot_row) <= 1
        )

    else:
        return None

    return robot_in_view


def use_shield(old_row_index, old_column_index, old_orientation):
    return (
        old_row_index,
        old_column_index,
        old_orientation,
        -10,
        False
    )


def get_collision(distance, robot_visibility, old_robot_row, old_robot_column, robot_orientation):
    # Rilevamento della collisione
    if robot_visibility and distance == 1 and go_forward(
            old_robot_row,
            old_robot_column,
            robot_orientation
    )[2]:
        return


# ============================================================
# TRAINING PARAMETERS
# ============================================================

discount_factor = 0.99
learning_rate = 0.3

episodes = 10000

# ============================================================
# TRAINING VARIABLES
# ============================================================

accumulated_reward = 0
number_of_steps = 0
i = 0
steps = 0
max_steps = 300
epsilon_start = 1.0
epsilon_end = 0.05

epsilon_decay = episodes
epsilon_decay_steps = 1000000

# Debug: stampa ogni transizione nei primi episodi; poi solo riepiloghi.
# Modalità di debug
DEBUG = True

# Se DEBUG=True, stampa i primi N episodi in dettaglio.
DEBUG_FIRST_EPISODES = 3

# Dopo i primi episodi, stampa ogni N passi.
DEBUG_EVERY_N_STEPS = 50

# Frequenza del riepilogo del training.
LOG_EVERY_EPISODES = 100

distance = 0
robot_visibility = 0
ped_visibility = 0
awareness = 0

visited_agent_cells = 0
visited_ped_cells = 0

# ============================================================
# DEBUG VARIABLES
# ============================================================

episode_rewards = []
episode_steps = []
episode_agent_coverage = []
episode_ped_coverage = []
episode_success = []
episode_q_mean = []
episode_q_max = []
episode_td_errors = []


class Tee:
    """Scrive contemporaneamente sul terminale e sul file di debug."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def train():
    global_step = 0

    # ============================================================
    # FASE 0: INIZIALIZZAZIONE DEL SISTEMA DI DEBUG
    # ============================================================
    # Crea un file di log per registrare le informazioni del training.
    # L'output viene inviato sia al terminale sia al file di debug.

    base_dir = os.path.dirname(os.path.abspath(__file__))
    debug_path = os.path.join(base_dir, "debug_qlearning.txt")

    debug_file = open(
        debug_path,
        "w",
        buffering=1,
        encoding="utf-8"
    )

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, debug_file)

    print(f"[DEBUG] Log del training: {debug_path}")
    print(f"[DEBUG] Task={task}, episodi={episodes}, max_steps={max_steps}")

    # ============================================================
    # FASE 1: CICLO PRINCIPALE DI APPRENDIMENTO
    # ============================================================
    # Il training è suddiviso in episodi.
    # In ogni episodio il robot affronta nuovamente il compito
    # partendo dalle posizioni iniziali configurate.
    #
    # Durante gli episodi, la Q-table viene aggiornata attraverso
    # le esperienze raccolte dall'agente nell'ambiente.

    for episode in range(episodes):
        remembered_robot_row = None
        remembered_robot_column = None
        td_errors = []
        # --------------------------------------------------------
        # FASE 1.1: RESET DELL'EPISODIO
        # --------------------------------------------------------
        # Ripristina le condizioni iniziali dell'ambiente.
        # Nel task 2 viene resettato anche lo stato interno
        # utilizzato per valutare il passing-side.

        print("\n==============================")
        print("Episodio:", episode + 1)
        print("==============================")

        # Inizializzazione degli orientamenti dei due agenti.
        robot_orientation = "UP"
        ped_orientation = "DOWN"
        awareness = 0.0

        # Memorizza quanti stati della Q-table esistono prima
        # dell'inizio dell'episodio, per misurare quanti ne
        # verranno aggiunti durante l'apprendimento.
        q_table_size_before = len(q_values)

        # Reset delle metriche dell'episodio.
        accumulated_reward = 0
        steps = 0
        # Memorizza l'azione eseguita al passo precedente.
        previous_action = None

        # Ripristino delle posizioni iniziali.
        robot_row, robot_column = getStartingRobotLocation()

        ped_row, ped_column = getStartingPedestrianLocation()

        # Inizializzazione della matrice di copertura per il task 1.
        if task == 1:

            cov.coverage_matrix.fill(0)

            # Registra la posizione iniziale del pedone.
            update_coverage_matrix(ped_row, ped_column, -1)
            update_coverage_matrix(robot_row, robot_column, 1)

        # ========================================================
        # FASE 2: INTERAZIONE CON L'AMBIENTE
        # ========================================================
        # Il ciclo continua finché:
        # - il compito non viene completato;
        # - non viene raggiunto il limite massimo di passi;
        # - non sono disponibili azioni legali.

        while True:

            if steps >= max_steps:
                done = False
                break

            # Resto della logica dell'episodio

            done = False

            # ====================================================
            # FASE 2.1: OSSERVAZIONE DELL'AMBIENTE
            # ====================================================
            # Il simulatore calcola la visibilità dei due agenti.
            #
            # Il robot può utilizzare le informazioni sul pedone
            # soltanto quando questo è visibile.
            #
            # Quando il pedone non è visibile:
            # - la distanza osservata viene impostata a infinito;
            # - l'orientamento osservato del pedone è sconosciuto.
            #
            # Nota: il simulatore conosce comunque lo stato reale
            # del pedone per calcolare le condizioni di visibilità.

            robot_visibility = get_robot_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            # Costruzione delle informazioni osservabili dal robot.
            if robot_visibility:

                ped_visibility = get_pedestrian_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    ped_orientation
                )

                distance = calculate_distance(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    robot_orientation
                )

                observed_ped_orientation = ped_orientation

            else:

                distance = float("inf")
                observed_ped_orientation = None
                ped_visibility = False

            # ====================================================
            # FASE 2.2: COSTRUZIONE DELLO STATO CORRENTE
            # ====================================================
            # Lo stato viene utilizzato dall'agente per consultare
            # la Q-table e scegliere l'azione successiva.
            #
            # Comprende:
            # - distanza osservata dal pedone;
            # - orientamento del robot;
            # - orientamento osservato del pedone;
            # - visibilità del pedone da parte del robot.

            front_cell_visited = False

            if task == 1:
                front_cell_visited = is_front_cell_visited(
                    robot_row,
                    robot_column,
                    robot_orientation
                )

            state = build_state(distance, robot_orientation, observed_ped_orientation, robot_visibility, ped_visibility, awareness, front_cell_visited, robot_row, robot_column)

            # ====================================================
            # FASE 2.3: AGGIORNAMENTO DEL PARAMETRO EPSILON
            # ====================================================
            # Epsilon controlla il compromesso tra:
            #
            # EXPLORATION:
            # scegliere un'azione casuale per esplorare l'ambiente.
            #
            # EXPLOITATION:
            # scegliere l'azione con il valore Q più elevato.
            #
            # Durante il training epsilon diminuisce gradualmente,
            # riducendo l'esplorazione e favorendo lo sfruttamento
            # delle conoscenze acquisite.

            # epsilon = decay_epsilon(
            #     episode,
            #     epsilon_decay,
            #     min_epsilon=epsilon_end
            # )

            epsilon = decay_epsilon_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)

            # Determina se stampare informazioni dettagliate
            # relative al passo corrente.
            debug_step = DEBUG and (
                episode < DEBUG_FIRST_EPISODES or steps % DEBUG_EVERY_N_STEPS == 0
            )

            if debug_step:

                print(
                    f"\n[DEBUG][EP {episode + 1}][STEP {steps}] "
                    f"state={state}, epsilon={epsilon:.3f}, "
                    f"pos_robot=({robot_row},{robot_column}), "
                    f"orientamento={robot_orientation}, "
                    f"pos_pedone=({ped_row},{ped_column}), "
                    f"distanza={distance}, "
                    f"vis_robot={robot_visibility}, "
                    f"vis_pedone={ped_visibility}"
                )

            # ====================================================
            # FASE 3: SELEZIONE DELL'AZIONE
            # ====================================================
            # L'agente seleziona un'azione tramite una strategia
            # epsilon-greedy.
            #
            # Ogni azione combina:
            # - un movimento del robot;
            # - una modalità di comunicazione.
            #
            # Vengono considerate soltanto le combinazioni
            # associate a movimenti legali nella situazione attuale.

            action_index = get_action(
                state,
                epsilon,
                robot_row,
                robot_column,
                robot_orientation
            )

            if action_index is None:
                break

            if debug_step:

                print(
                    f"[DEBUG][AWARENESS] "
                    f"awareness={awareness} "
                )

            movement_action = actions[action_index]

            if debug_step:

                print(
                    f"[DEBUG][AZIONE] "
                    f"azione={movement_action} "
                )

            # Registra la posizione del pedone nella matrice
            # di copertura, se necessario.
            if task == 1:
                if robot_visibility:
                    ped_row_visibile = ped_row
                    ped_column_visibile = ped_column

                    update_coverage_matrix(ped_row_visibile, ped_column_visibile, -1)

            # ====================================================
            # FASE 4: ESECUZIONE DELL'AZIONE
            # ====================================================
            # Memorizza lo stato precedente del robot.
            # Queste informazioni saranno utilizzate per valutare
            # la transizione e calcolare la ricompensa.

            # Salva lo stato precedente e le osservazioni.
            old_robot_row = robot_row
            old_robot_column = robot_column
            old_ped_row = ped_row
            old_ped_column = ped_column
            old_robot_orientation = robot_orientation

            previous_robot_visibility = robot_visibility
            previous_distance = distance

            # ====================================================
            # FASE 4.1: CAMBIO DI STATO DOPO L'AZIONE
            # ====================================================
            # Calcola la posizione candidata senza modificare ancora
            # la posizione effettiva del robot.
            (
                robot_row,
                robot_column,
                robot_orientation,
                robot_moved
            ) = get_next_position(
                robot_row,
                robot_column,
                robot_orientation,
                action_index
            )

            # ====================================================
            # FASE 4.2: CONTROLLO DELLA COLLISIONE COL PEDONE
            # ====================================================
            # Controlla se il movimento selezionato porterebbe il robot
            # nella cella attualmente occupata dal pedone.
            collision = (
                action_index == 0 and robot_moved and (robot_row, robot_column) == (ped_row, ped_column)
            )

            if collision:

                (
                    robot_row,
                    robot_column,
                    robot_orientation,
                    reward,
                    done
                ) = use_shield(
                    old_robot_row,
                    old_robot_column,
                    old_robot_orientation
                )

                robot_moved = False

            else:

                # Aggiorna la visibilità dopo il movimento.
                robot_visibility = get_robot_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    robot_orientation
                )

                if robot_visibility:

                    ped_visibility = get_pedestrian_visibility(
                        robot_row,
                        robot_column,
                        ped_row,
                        ped_column,
                        ped_orientation
                    )

                    distance = calculate_distance(
                        robot_row,
                        robot_column,
                        ped_row,
                        ped_column,
                        robot_orientation
                    )
                else:
                    distance = float("inf")

                old_awareness = awareness
                ped_visibility_aware = get_pedestrian_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    ped_orientation
                )

                awareness = get_awareness(ped_visibility_aware, robot_visibility, action_index, awareness)

                # ====================================================
                # FASE 5: CALCOLO DELLA REWARD
                # ====================================================
                # Calcola la reward del task.
                reward, done = get_reward(
                    old_robot_row,
                    old_robot_column,
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    previous_distance,
                    distance,
                    action_index,
                    previous_action,
                    previous_robot_visibility,
                    robot_visibility,
                    robot_orientation,
                    robot_moved,
                    ped_visibility,
                    old_awareness,
                    awareness
                )

            # Memorizza l'azione corrente per il passo successivo.
            previous_action = action_index

            # Accumula la ricompensa per calcolare il risultato
            # complessivo dell'episodio.
            accumulated_reward += reward

            if debug_step:

                print(
                    f"[DEBUG][TRANSIZIONE] "
                    f"robot=({old_robot_row},{old_robot_column}) "
                    f"-> ({robot_row},{robot_column}), "
                    f"orientamento={old_robot_orientation}"
                    f"->{robot_orientation}, "
                    f"robot_moved={robot_moved}, collision={collision}, "
                    f"reward={reward:.3f}, done={done}"
                )

                if not collision:

                    print(
                        f"[DEBUG][REWARD] "
                        f"task={reward:.3f}"
                    )

            # ====================================================
            # FASE 6: AGGIORNAMENTO DELL'AMBIENTE
            # ====================================================
            # La transizione del robot è stata eseguita.
            # Si aggiorna la matrice di copertura e si esegue
            # il movimento del pedone.
            #
            # Il movimento del pedone modifica lo stato reale
            # dell'ambiente e può influenzare la prossima
            # osservazione del robot.

            if task == 1:

                update_coverage_matrix(
                    robot_row,
                    robot_column,
                    1
                )

            # ====================================================
            # FASE 6.1: MOVIMENTO DEL PEDONE
            # ====================================================
            # Il pedone sceglie una nuova posizione tra quelle
            # consentite, evitando i limiti della griglia e
            # la cella occupata dal robot.
            #
            # Questa transizione modifica lo stato dell'ambiente
            # prima della costruzione del prossimo stato.

            ped_row, ped_column, ped_orientation, ped_moved, remembered_robot_row, remembered_robot_column = (
                update_pedestrian_position(
                    ped_row,
                    ped_column,
                    remembered_robot_row,
                    remembered_robot_column,
                    robot_row,
                    robot_column,
                    ped_orientation,
                    awareness
                )
            )

            # ========================================================
            # VERIFICA DEL COMPLETAMENTO DEL TASK
            # ========================================================

            if task == 1:

                next_robot_visibility = get_robot_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    robot_orientation
                )

                if next_robot_visibility:
                    update_coverage_matrix(
                        ped_row,
                        ped_column,
                        -1
                    )

                # Verifica il completamento sulla matrice aggiornata.
                done = is_coverage_completed()

                if done:
                    reward += 20
                    accumulated_reward += 20

                    if DEBUG:
                        print(">>> COVERAGE COMPLETATA!")

            # ====================================================
            # FASE 7: COSTRUZIONE DEL NUOVO STATO
            # ====================================================
            # Dopo aver aggiornato robot e pedone, si calcola
            # la nuova osservazione dell'ambiente.
            #
            # La nuova osservazione sarà utilizzata per valutare
            # il valore futuro delle azioni nella Q-table.
            #
            # Se il robot non vede il pedone, distanza e
            # orientamento osservato non devono rivelarne
            # direttamente lo stato reale.

            next_robot_visibility = get_robot_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            if next_robot_visibility:

                next_ped_visibility = get_pedestrian_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    ped_orientation
                )

                next_distance = calculate_distance(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    robot_orientation
                )

                next_observed_ped_orientation = ped_orientation

            else:

                next_ped_visibility = False
                next_distance = float("inf")
                next_observed_ped_orientation = None

            # NOTA:
            # Lo stato successivo deve avere la stessa struttura
            # dello stato corrente.
            #
            # Se lo stato corrente contiene quattro elementi,
            # anche next_state deve contenerne quattro.

            next_front_cell_visited = False

            if task == 1:
                next_front_cell_visited = is_front_cell_visited(
                    robot_row,
                    robot_column,
                    robot_orientation
                )

            next_state = build_state(
                next_distance,
                robot_orientation,
                next_observed_ped_orientation,
                next_robot_visibility,
                next_ped_visibility,
                awareness,
                next_front_cell_visited,
                robot_row,
                robot_column
            )

            steps += 1
            global_step += 1

            truncated = False

            # if task == 1 and not done and steps >= max_steps:

            #     coverage = round(float(percentage_coverage()), 3)

            #     penalty = -10.0 * (1.0 - coverage)

            #     reward += penalty
            #     accumulated_reward += penalty

            #     truncated = True

            #     if DEBUG:
            #         print(
            #             f">>> TIMEOUT: copertura non completata. "
            #             f"Copertura={coverage:.1%}, "
            #             f"penalità={penalty:.3f}"
            #         )

            epsilon = decay_epsilon_step(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)

            # ====================================================
            # FASE 8: AGGIORNAMENTO DELLA Q-TABLE
            # ====================================================
            # Applica la regola di aggiornamento del Q-learning:
            #
            # Q(s,a) <- Q(s,a) + alpha * [
            #     reward + gamma * max_a' Q(s',a') - Q(s,a)
            # ]
            #
            # alpha: learning rate.
            # gamma: discount factor.
            # reward: ricompensa ottenuta dalla transizione.
            # max Q(s',a'): miglior valore futuro stimato.
            #
            # Se lo stato è terminale, il valore futuro è zero.

            if done or truncated:
                next_q = 0.0
            else:
                next_legal_actions = get_legal_actions(
                    robot_row,
                    robot_column,
                    robot_orientation
                )

                if not next_legal_actions:
                    next_q = 0.0
                else:
                    next_q = np.max(
                        q_values[tuple(next_state)][next_legal_actions]
                    )

            # ----------------------------------------------------
            # FASE 8.1: CALCOLO DELL'ERRORE TEMPORALE (TD ERROR)
            # ----------------------------------------------------
            # Confronta la stima Q corrente con il target:
            #
            # target = reward + gamma * next_q
            #
            # TD error = target - Q(s,a)
            #
            # L'errore indica quanto la stima attuale differisce
            # dal valore suggerito dall'esperienza appena raccolta.

            old_q = q_values[tuple(state)][action_index]

            target_q = reward + discount_factor * next_q

            td_error = target_q - old_q

            # ----------------------------------------------------
            # FASE 8.2: AGGIORNAMENTO DEL VALORE Q
            # ----------------------------------------------------
            # Corregge il valore Q usando il learning rate.
            #
            # Un learning rate elevato produce aggiornamenti
            # più ampi; uno basso rende l'apprendimento graduale.

            new_q = old_q + learning_rate * td_error

            q_values[tuple(state)][action_index] = new_q

            td_errors.append(abs(td_error))

            if debug_step:

                print(
                    f"[DEBUG][Q-UPDATE] "
                    f"Q_old={old_q:.4f}, "
                    f"target={target_q:.4f}, "
                    f"next_q={next_q:.4f}, "
                    f"TD_error={td_error:.4f}, "
                    f"Q_new={new_q:.4f}, "
                    f"stati_Q={len(q_values)}"
                )

            # ====================================================
            # FASE 9: CONTROLLO DELLA TERMINAZIONE
            # ====================================================
            # Se il compito è stato completato, l'episodio termina.
            # Altrimenti, il robot continua a interagire
            # con l'ambiente.
            #
            # L'aggiornamento Q è già stato eseguito anche
            # nell'ultima transizione.

            if done:

                if debug_step:

                    print(
                        f"[DEBUG][TERMINAZIONE] "
                        f"Episodio {episode + 1} terminato "
                        f"al passo {steps}; "
                        f"reward cumulativa parziale="
                        f"{accumulated_reward:.3f}"
                    )

                break

        # ========================================================
        # FASE 10: VALUTAZIONE DELL'EPISODIO
        # ========================================================
        # Al termine dell'episodio si registrano le metriche
        # utili a monitorare l'andamento dell'apprendimento.
        #
        # Queste metriche consentono di analizzare:
        # - reward cumulativa;
        # - numero di passi;
        # - successo del compito;
        # - copertura della griglia;
        # - valori appresi nella Q-table.

        if task == 1:

            visited_agent_cells = np.sum(cov.coverage_matrix == 1)
            visited_ped_cells = np.sum(cov.coverage_matrix == -1)

        else:

            visited_agent_cells = 0
            visited_ped_cells = 0

        # Estrae i massimi valori Q degli stati che hanno
        # almeno un valore diverso da zero.
        q_values_used = [
            np.max(values)
            for values in q_values.values()
            if np.any(values != 0)
        ]

        if q_values_used:

            mean_q = np.mean(q_values_used)
            max_q = np.max(q_values_used)

        else:

            mean_q = 0
            max_q = 0

        # Stampa il riepilogo dell'episodio.
        if (
            (episode + 1) % LOG_EVERY_EPISODES == 0
            or episode == 0
            or episode == episodes - 1
        ):
            print(
                f"[TRAINING] "
                f"Episodio={episode + 1}/{episodes} | "
                f"Reward={accumulated_reward:.2f} | "
                f"Passi={steps}/{max_steps} | "
                f"Epsilon={epsilon:.3f} | "
                f"Stati Q={len(q_values)} | "
                f"Nuovi stati={len(q_values) - q_table_size_before} | "
                f"Successo={done}"
            )

        # Salva le metriche per l'analisi successiva.
        episode_rewards.append(accumulated_reward)
        episode_steps.append(steps)
        episode_agent_coverage.append(visited_agent_cells)
        episode_ped_coverage.append(visited_ped_cells)
        episode_success.append(done)
        episode_q_mean.append(mean_q)
        episode_q_max.append(max_q)

        # Una sola media dell'errore TD per episodio
        mean_td_error = np.mean(td_errors) if td_errors else 0.0
        episode_td_errors.append(mean_td_error)

    # ============================================================
    # SALVATAGGIO DELLE METRICHE DI TRAINING
    # ============================================================

    metrics_path = os.path.join(base_dir, "training_metrics.npz")

    np.savez(
        metrics_path,
        episode_rewards=np.array(episode_rewards),
        episode_success=np.array(episode_success, dtype=bool),
        episode_steps=np.array(episode_steps),
        episode_td_errors=np.array(episode_td_errors),
        episode_agent_coverage=np.array(episode_agent_coverage),
        episode_ped_coverage=np.array(episode_ped_coverage),
        episode_q_mean=np.array(episode_q_mean),
        episode_q_max=np.array(episode_q_max)
    )

    print(f"Metriche salvate in: {metrics_path}")

    # ============================================================
    # FASE 11: CONCLUSIONE DEL TRAINING E SALVATAGGIO
    # ============================================================
    # Terminati tutti gli episodi, salva la Q-table e i metadati
    # necessari per riutilizzarla durante la fase di test.

    print("Training complete!")

    checkpoint = {
        "q_values": dict(q_values),
        "rows": rows,
        "columns": columns,
        "actions": actions,
        "task": task
    }

    checkpoint_path = os.path.join(base_dir, "q_table.pkl")

    with open(checkpoint_path, "wb") as file:
        pickle.dump(checkpoint, file)

    print(f"Q-table salvata in: {checkpoint_path}")
    print(f"Numero di stati salvati: {len(q_values)}")

    print(f"[DEBUG] Log completato: {debug_path}")

    # Ripristina l'output standard e chiude il file di debug.
    sys.stdout = original_stdout
    debug_file.close()


if __name__ == "__main__":
    print("Avvio del training...", flush=True)
    configure_training()
    train()
