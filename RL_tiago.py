import numpy as np
import time
from collections import defaultdict
# import rclpy

from coverage import (
    coverage_matrix,
    update_coverage_matrix,
    is_coverage_completed
)

from passing_side import (
    reward_passing_side,
    reset_passing_side
)

# from rclpy.node import Node
# from geometry_msgs.msg import TwistStamped
# import time
import random
# from gazebo_msgs.srv import SetEntityState

# ============================================================
# ENVIRONMENT
# ============================================================

rows = 5
columns = 4

actions = [
    "forward",
    "left_forward",
    "right_forward",
    "turn_left",
    "turn_right"
]

communication_actions = [
    "none",
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
n_q_actions = len(actions) * len(communication_actions)

q_values = defaultdict(
    lambda: np.zeros(n_q_actions)
)

# Values used to identify safe/unsafe actions
safe_actions = np.zeros((rows, columns, 4))


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


# ============================================================
# TERMINAL STATE
# ============================================================
def is_terminal_state(ped_row, ped_column, robot_visibility, distance, robot_orientation, old_robot_column, new_robot_column, old_robot_row, new_robot_row, ped_visibility):

    if task == 1:

        covered_cells = coverage_matrix != 0

        # La cella del pedone è inclusa dalla coverage
        covered_cells[ped_row, ped_column] = True

        print("\n========== TERMINAL CHECK ==========")
        print("Coverage matrix:")
        print(coverage_matrix)

        print("Celle coperte:")
        print(np.sum(covered_cells))

        print("Celle totali da coprire:", coverage_matrix.size - 1)

        print("Celle visitate dal robot:", np.sum(coverage_matrix == 1))

        print("Celle visitate dal pedone:", np.sum(coverage_matrix == -1))

        coverage_completed = np.all(covered_cells)

        print("Tutte le celle sono coperte?:", coverage_completed)

        if coverage_completed:
            print(">>> TERMINAL STATE = TRUE")
            print(">>> COVERAGE COMPLETATA!")
            return True

        print(">>> TERMINAL STATE = FALSE")

        return False

    elif task == 2:
        return is_passing_completed(
            distance,
            robot_visibility,
            robot_orientation,
            old_robot_column,
            new_robot_column,
            old_robot_row,
            new_robot_row,
            ped_visibility
        )


def go_forward(row_index, column_index, orientation, ped_row, ped_column):

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

    # Controllo dei limiti della griglia
    if not (0 <= new_row < rows and 0 <= new_column < columns):
        return row_index, column_index, False, False

    # Rilevamento della collisione
    if (new_row, new_column) == (ped_row, ped_column):
        return row_index, column_index, False, True

    # Movimento consentito
    return new_row, new_column, True, False


# ============================================================
# TURN LEFT
# ============================================================

def turn_left(
    row_index,
    column_index,
    robot_orientation,
    ped_row,
    ped_column
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
    row_index,
    column_index,
    robot_orientation,
    ped_row,
    ped_column
):
    orientations = ["UP", "RIGHT", "DOWN", "LEFT"]

    current_index = orientations.index(robot_orientation)

    new_orientation = orientations[
        (current_index + 1) % len(orientations)
    ]

    # La rotazione sul posto è sempre consentita.
    return new_orientation, True


def get_communication_reward(
    communication,
    robot_visibility,
    ped_visibility,
    distance
):
    if communication == "none":
        return 0.0

    if (
        communication == "acoustic"
        and robot_visibility
        and not ped_visibility
    ):
        return 0.5

    elif (
        communication == "visual"
        and robot_visibility
        and ped_visibility
        and distance <= 2
    ):
        return 0.5

    elif (
        communication == "projection"
        and not robot_visibility
        and not ped_visibility
        and distance > 2
    ):
        return 0.5

    return -0.5


# ============================================================
# REWARD FUNCTION
# ============================================================
def get_task_reward(
    old_robot_row,
    old_robot_column,
    new_robot_row,
    new_robot_column,
    distance,
    action,
    previous_robot_visibility,
    robot_visibility,
    robot_orientation,
    ped_row,
    ped_column,
    ped_visibility,
    moved
):
    # Inizializzazione
    reward = 0
    done = False

    passing_action_names = {
        0: "forward",
        1: "left_forward",
        2: "right_forward",
        3: "turn_left",
        4: "turn_right"
    }

    # ========================================================
    # REWARD GENERICI
    # ========================================================

    # Collisione con il pedone
    if (
        distance == 1 and robot_visibility and go_forward(
            old_robot_row,
            old_robot_column,
            robot_orientation,
            ped_row,
            ped_column
        )[2]
    ):
        reward = -10

    # Risultati riferiti all'azione realmente eseguita
    turn_left_result = action in (1, 3)
    turn_right_result = action in (2, 4)

    # Solo forward, left_forward e right_forward prevedono avanzamento
    go_forward_result = moved if action in (0, 1, 2) else False

    if (old_robot_row == new_robot_row and old_robot_column == new_robot_column):
        reward = -0.5

    if (
        not turn_left_result
        and not turn_right_result
        and not go_forward_result
    ):
        reward = -1

    # ========================================================
    # TASK 1: COVERAGE - REWARD SHAPING
    # ========================================================
    if task == 1:

        # Nuova cella visitata dal robot
        if coverage_matrix[new_robot_row, new_robot_column] == 0:
            reward = 1

        # Cella già visitata o occupata dal pedone
        else:
            reward = -0.5

    # ========================================================
    # TASK 2: PASSING-SIDE
    # ========================================================
    elif task == 2:

        # if robot_observation and go_forward_result:
        #     return 0.5, False

        executed_action = passing_action_names[action]

        reward, done = reward_passing_side(
            previous_robot_visibility,
            robot_visibility,
            distance,
            turn_left_result,
            turn_right_result,
            go_forward_result,
            executed_action
        )

        print(
            f"[PASSING DEBUG] "
            f"azione={executed_action}, "
            f"osservazione_precedente={previous_robot_visibility}, "
            f"osservazione_attuale={robot_visibility}, "
            f"rotazione_sinistra={turn_left_result}, "
            f"rotazione_destra={turn_right_result}, "
            f'avanzamento={go_forward_result}, '
            f"done={done}"
        )

    return reward, done


# ============================================================
# INITIAL STATE
# ============================================================
def getStartingRobotLocation():

    row_index = start_row
    column_index = start_column

    return row_index, column_index


# ============================================================
# OBSTACLE STATE
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


def can_move_forward(row, column, orientation):
    """Controlla se la cella davanti al robot è dentro la griglia."""

    next_row = row
    next_column = column

    if orientation == "UP":
        next_row -= 1
    elif orientation == "DOWN":
        next_row += 1
    elif orientation == "LEFT":
        next_column -= 1
    elif orientation == "RIGHT":
        next_column += 1

    return (
        0 <= next_row < rows
        and 0 <= next_column < columns
    )


def get_legal_actions(
    robot_row,
    robot_column,
    robot_orientation,
    ped_row,
    ped_column
):
    legal_actions = []

    for action_index in range(len(actions)):

        (
            new_row,
            new_column,
            new_orientation,
            moved,
            collision
        ) = get_next_state(
            robot_row,
            robot_column,
            robot_orientation,
            ped_row,
            ped_column,
            action_index
        )

        # Le azioni di avanzamento devono muovere il robot,
        # a meno che non si tratti di una collisione.
        if action_index in (0, 1, 2) and not moved and not collision:
            continue

        # Le rotazioni sul posto restano ammesse solo se
        # nella nuova direzione è possibile avanzare.
        if action_index in (3, 4):
            if not can_move_forward(
                robot_row,
                robot_column,
                new_orientation
            ):
                continue

        # Verifica i limiti della griglia.
        if not (
            0 <= new_row < rows
            and 0 <= new_column < columns
        ):
            continue

        # Non escludere le azioni che causano collisione:
        # il robot deve poterle selezionare e imparare
        # dalla penalità.
        if (
            (new_row, new_column) == (ped_row, ped_column)
            and not collision
        ):
            continue

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
    robot_orientation,
    ped_row,
    ped_column
):
    legal_movements = get_legal_actions(
        robot_row,
        robot_column,
        robot_orientation,
        ped_row,
        ped_column
    )

    if not legal_movements:
        return None

    # Combina ogni movimento legale con ogni modalità
    # di comunicazione.
    possible_actions = [
        encode_action(movement, communication)
        for movement in legal_movements
        for communication in range(len(communication_actions))
    ]

    state_key = tuple(state)

    # Esplorazione
    if np.random.random() < epsilon:
        return int(np.random.choice(possible_actions))

    # Sfruttamento
    legal_q_values = q_values[state_key][possible_actions]

    best_index = int(np.argmax(legal_q_values))

    return int(possible_actions[best_index])


def encode_action(movement_index, communication_index):
    return (
        movement_index * len(communication_actions)
        + communication_index
    )


def decode_action(action_index):
    movement_index = action_index // len(communication_actions)
    communication_index = action_index % len(communication_actions)

    return movement_index, communication_index


# ============================================================
# STATE TRANSITION
# ============================================================

def get_next_state(
    robot_row,
    robot_column,
    robot_orientation,
    ped_row,
    ped_column,
    action
):
    new_robot_row = robot_row
    new_robot_column = robot_column
    new_orientation = robot_orientation
    moved = False
    collision = False

    if action == 0:  # forward
        (
            new_robot_row,
            new_robot_column,
            moved,
            collision
        ) = go_forward(
            robot_row,
            robot_column,
            robot_orientation,
            ped_row,
            ped_column
        )

    elif action == 1:  # left_forward
        new_orientation, _ = turn_left(
            robot_row,
            robot_column,
            robot_orientation,
            ped_row,
            ped_column
        )

        (
            new_robot_row,
            new_robot_column,
            moved,
            collision
        ) = go_forward(
            robot_row,
            robot_column,
            new_orientation,
            ped_row,
            ped_column
        )

    elif action == 2:  # right_forward
        new_orientation, _ = turn_right(
            robot_row,
            robot_column,
            robot_orientation,
            ped_row,
            ped_column
        )

        (
            new_robot_row,
            new_robot_column,
            moved,
            collision
        ) = go_forward(
            robot_row,
            robot_column,
            new_orientation,
            ped_row,
            ped_column
        )

    elif action == 3:  # turn_left
        new_orientation, _ = turn_left(
            robot_row,
            robot_column,
            robot_orientation,
            ped_row,
            ped_column
        )

    elif action == 4:  # turn_right
        new_orientation, _ = turn_right(
            robot_row,
            robot_column,
            robot_orientation,
            ped_row,
            ped_column
        )

    return (
        new_robot_row,
        new_robot_column,
        new_orientation,
        moved,
        collision
    )


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


# DA CAMBIARE
def control_illegals_states(
    current_row_index,
    current_column_index
):

    illegal_actions = []

    for i in range(len(actions)):

        if safe_actions[
            current_row_index,
            current_column_index,
            i
        ] < 0:

            illegal_actions.append(i)

    return illegal_actions


# ============================================================
# UPDATE PEDESTRIAN POSITION
# ============================================================
def update_pedestrian_position(
    ped_row,
    ped_column,
    robot_row,
    robot_column,
    ped_orientation
):
    possible_positions = [
        (ped_row - 1, ped_column, "UP"),
        (ped_row + 1, ped_column, "DOWN"),
        (ped_row, ped_column + 1, "RIGHT"),
        (ped_row, ped_column - 1, "LEFT")
    ]

    possible_positions = [
        position
        for position in possible_positions
        if (
            0 <= position[0] < rows and 0 <= position[1] < columns and position[:2] != (robot_row, robot_column)
        )
    ]

    if not possible_positions:
        return ped_row, ped_column, ped_orientation

    new_row, new_column, new_orientation = random.choice(
        possible_positions
    )

    return new_row, new_column, new_orientation


def calculate_distance(
    robot_row,
    robot_column,
    pedestrian_row,
    pedestrian_column,
    robot_orientation
):
    if (
        robot_row == pedestrian_row and robot_orientation in ("RIGHT", "LEFT")
    ):
        return abs(robot_column - pedestrian_column)

    if (
        robot_column == pedestrian_column and robot_orientation in ("UP", "DOWN")
    ):
        return abs(robot_row - pedestrian_row)

    return float("inf")


def visibility(
    robot_orientation,
    ped_orientation,
    robot_row,
    robot_column,
    ped_row,
    ped_column
):
    robot_visibility = False
    ped_visibility = False

    # ========================================================
    # CASO 1: ROBOT E PEDONE SULLA STESSA COLONNA
    # ========================================================

    if robot_column == ped_column:

        # Pedone sopra il robot
        if robot_row > ped_row:

            # Il robot guarda verso l'alto
            if robot_orientation == "UP":
                robot_visibility = True

            # Il pedone guarda verso il basso
            if ped_orientation == "DOWN":
                ped_visibility = True

        # Pedone sotto il robot
        elif robot_row < ped_row:

            # Il robot guarda verso il basso
            if robot_orientation == "DOWN":
                robot_visibility = True

            # Il pedone guarda verso l'alto
            if ped_orientation == "UP":
                ped_visibility = True

    # ========================================================
    # CASO 2: ROBOT E PEDONE SULLA STESSA RIGA
    # ========================================================

    elif robot_row == ped_row:

        # Pedone a sinistra del robot
        if robot_column > ped_column:

            # Il robot guarda verso sinistra
            if robot_orientation == "LEFT":
                robot_visibility = True

            # Il pedone guarda verso destra
            if ped_orientation == "RIGHT":
                ped_visibility = True

        # Pedone a destra del robot
        elif robot_column < ped_column:

            # Il robot guarda verso destra
            if robot_orientation == "RIGHT":
                robot_visibility = True

            # Il pedone guarda verso sinistra
            if ped_orientation == "LEFT":
                ped_visibility = True

    return robot_visibility, ped_visibility


def use_shield(old_row_index, old_column_index, old_orientation):
    return (
        old_row_index,
        old_column_index,
        old_orientation,
        -10,
        False
    )


# ============================================================
# TRAINING PARAMETERS
# ============================================================

discount_factor = 0.9
learning_rate = 0.9

episodes = 2000

# ============================================================
# TRAINING VARIABLES
# ============================================================

accumulated_reward = 0
number_of_steps = 0
i = 0
steps = 0
max_steps = 500
epsilon = 0.95
epsilon_min = 0.05

epsilon_decay = episodes

distance = 0
robot_visibility = 0
ped_visibility = 0

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


def train():
    debug_file = open("debug_qlearning.txt", "w", buffering=1)

    for episode in range(episodes):

        if task == 2:
            reset_passing_side()

        print("\n==============================")
        print("Episodio:", episode + 1)
        print("==============================")

        # Orientations
        robot_orientation = "UP"
        ped_orientation = "DOWN"

        # Q-table size initialization
        q_table_size_before = len(q_values)

        accumulated_reward = 0

        # Steps counter reset
        steps = 0

        # Starting robot position
        robot_row, robot_column = getStartingRobotLocation()

        # Starting pedestrian position
        ped_row, ped_column = getStartingPedestrianLocation()

        if task == 1:
            # coverage matrix initialization
            coverage_matrix.fill(0)

            # Contrassegna la cella del pedone
            if coverage_matrix[ped_row, ped_column] == 0:
                coverage_matrix[ped_row, ped_column] = -1

            # Registra la cella iniziale del robot
            update_coverage_matrix(robot_row, robot_column)

        # --------------------------------------------------------
        # EPISODE
        # --------------------------------------------------------

        while True:

            steps += 1

            # Limite al numero massimo di step
            # if steps >= max_steps:
            #     break

            done = False

            # ==========================================
            # 1. STATO CORRENTE
            # ==========================================
            distance = calculate_distance(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            robot_visibility, ped_visibility = visibility(
                robot_orientation,
                ped_orientation,
                robot_row,
                robot_column,
                ped_row,
                ped_column
            )

            observed_ped_orientation = (
                ped_orientation if robot_visibility else None
            )

            previous_robot_visibility = robot_visibility

            state = (
                distance,
                robot_orientation,
                observed_ped_orientation,
                robot_visibility,
                ped_visibility
            )

            # ==========================================
            # 2. SCELTA AZIONE
            # ==========================================
            epsilon = decay_epsilon(episode, epsilon_decay, min_epsilon=epsilon_min)
            print("epsilon:", epsilon)

            action_index = get_action(
                state,
                epsilon,
                robot_row,
                robot_column,
                robot_orientation,
                ped_row,
                ped_column
            )

            if action_index is None:
                break

            # Mantieni l'indice originale dell'azione combinata (0-19)
            combined_action_index = action_index

            # Separa movimento e comunicazione
            movement_index, communication_index = decode_action(
                combined_action_index
            )

            # Ottieni i nomi delle azioni
            movement_action = actions[movement_index]

            communication_action = communication_actions[
                communication_index
            ]

            # if len(obstacle_path) > 1:
            #     obstacle_path_rows, obstacle_path_columns = zip(*obstacle_path)

            #     ax1.plot(
            #         obstacle_path_columns,
            #         obstacle_path_rows,
            #         color="black",
            #         linewidth=2
            #     )

            if task == 1:
                if coverage_matrix[ped_row, ped_column] == 0:
                    coverage_matrix[ped_row, ped_column] = -1

            if action_index is None:
                break

            # Salvo posizione precedente
            old_row_index = robot_row
            old_column_index = robot_column
            old_orientation = robot_orientation

            # Esegui l'azione
            (
                new_next_row,
                new_next_column,
                new_orientation,
                moved,
                collision
            ) = get_next_state(
                robot_row,
                robot_column,
                robot_orientation,
                ped_row,
                ped_column,
                movement_index
            )

            if collision:
                (
                    new_next_row,
                    new_next_column,
                    new_orientation,
                    reward,
                    done
                ) = use_shield(
                    old_row_index,
                    old_column_index,
                    old_orientation
                )

            else:

                # Calcola visibilità e distanza dopo il movimento
                new_robot_visibility, new_ped_visibility = visibility(
                    new_orientation,
                    ped_orientation,
                    new_next_row,
                    new_next_column,
                    ped_row,
                    ped_column
                )

                new_distance = calculate_distance(
                    new_next_row,
                    new_next_column,
                    ped_row,
                    ped_column,
                    new_orientation
                )

                reward_task, done = get_task_reward(
                    old_row_index,
                    old_column_index,
                    new_next_row,
                    new_next_column,
                    distance,
                    movement_index,
                    previous_robot_visibility,
                    new_robot_visibility,
                    new_orientation,
                    ped_row,
                    ped_column,
                    new_ped_visibility,
                    moved
                )

                reward_comm = get_communication_reward(
                    communication_action,
                    new_robot_visibility,
                    new_ped_visibility,
                    new_distance
                )

                reward = reward_task + reward_comm

            # Aggiorna lo stato effettivo del robot
            robot_row = new_next_row
            robot_column = new_next_column
            robot_orientation = new_orientation

            row_index = robot_row
            column_index = robot_column

            # ==========================================
            # 6. AGGIORNO COVERAGE
            # ==========================================
            if task == 1:
                update_coverage_matrix(
                    row_index,
                    column_index
                )

            # ==========================================
            # 6.1. SCELTA AZIONE PEDONE
            # ==========================================
            ped_row, ped_column, ped_orientation = update_pedestrian_position(
                ped_row,
                ped_column,
                robot_row,
                robot_column,
                ped_orientation
            )

            # Registra anche la posizione attuale del pedone
            if task == 1:
                if coverage_matrix[ped_row, ped_column] == 0:
                    coverage_matrix[ped_row, ped_column] = -1

                # Controlla la coverage dopo gli aggiornamenti
                done = is_coverage_completed()

                if done:
                    reward += 10
                    accumulated_reward += 10

                    print(">>> COVERAGE COMPLETATA!")
                    print(coverage_matrix)

            # ==========================================
            # 7. CONTROLLO TERMINALE
            # ==========================================

            distance = calculate_distance(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            robot_visibility, ped_visibility = visibility(
                robot_orientation,
                ped_orientation,
                robot_row,
                robot_column,
                ped_row,
                ped_column
            )

            # ==========================================
            # 8. NUOVO STATO
            # ==========================================

            next_robot_visibility, next_ped_visibility = visibility(
                robot_orientation,
                ped_orientation,
                robot_row,
                robot_column,
                ped_row,
                ped_column
            )

            next_distance = calculate_distance(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            next_observed_ped_orientation = (
                ped_orientation if next_robot_visibility else None
            )

            next_state = (
                next_distance,
                robot_orientation,
                next_observed_ped_orientation,
                next_robot_visibility,
                next_ped_visibility
            )

            # ==========================================
            # 9. Q-LEARNING UPDATE
            # ==========================================

            if done:
                next_q = 0.0

            else:
                next_legal_movements = get_legal_actions(
                    robot_row,
                    robot_column,
                    robot_orientation,
                    ped_row,
                    ped_column
                )

                next_legal_actions = [
                    encode_action(movement, communication)
                    for movement in next_legal_movements
                    for communication in range(len(communication_actions))
                ]

                if not next_legal_actions:
                    next_q = 0.0
                else:
                    next_q = np.max(
                        q_values[tuple(next_state)][next_legal_actions]
                    )

            # Aggiornamento SEMPRE eseguito, anche se done=True
            q_values[tuple(state)][combined_action_index] += learning_rate * (
                reward
                + discount_factor * next_q
                - q_values[tuple(state)][combined_action_index]
            )

            # ==========================================
            # 10. TERMINA EPISODIO
            # ==========================================

            if done:
                break

            print("End of episode!")

        # ==========================================
        # EPISODE DEBUG
        # ==========================================
        if task == 1:
            visited_agent_cells = np.sum(coverage_matrix == 1)
            visited_ped_cells = np.sum(coverage_matrix == -1)
        else:
            visited_agent_cells = 0
            visited_ped_cells = 0

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

        print("Stati conosciuti:", len(q_values))
        print(
            "Nuovi stati:",
            len(q_values) - q_table_size_before
        )

        episode_rewards.append(accumulated_reward)
        episode_steps.append(steps)
        episode_agent_coverage.append(visited_agent_cells)
        episode_ped_coverage.append(visited_ped_cells)
        episode_success.append(done)
        episode_q_mean.append(mean_q)
        episode_q_max.append(max_q)

        if (episode + 1) % 10 == 0:

            debug_file.write(
                f"Episode: {episode + 1}\n"
                f"Reward totale: {accumulated_reward}\n"
                f"Steps: {steps}\n"
                f"Celle visitate dall'agente: {visited_agent_cells}\n"
                f"Celle visitate dal pedone: {visited_ped_cells}\n"
                f"Coverage completata: {done}\n"
                f"Q medio: {mean_q}\n"
                f"Q massimo: {max_q}\n"
                f"Epsilon: {epsilon}\n"
                f"Stati conosciuti: {len(q_values)}\n"
                f"Nuovi stati: {len(q_values) - q_table_size_before}\n"
                f"{'-' * 50}\n"
            )

    # ============================================================
    # TRAINING COMPLETED
    # ============================================================

    print("Training complete!")

    # Salvataggio della Q-table e dei metadati
    import os
    import pickle

    checkpoint = {
        "q_values": dict(q_values),
        "rows": rows,
        "columns": columns,
        "actions": actions,
        "communication_actions": communication_actions,
        "task": task
    }

    # Salva il file nella stessa cartella di RL_tiago.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, "q_table.pkl")

    with open(checkpoint_path, "wb") as file:
        pickle.dump(checkpoint, file)

    print(f"Q-table salvata in: {checkpoint_path}")
    print(f"Numero di stati salvati: {len(q_values)}")

    debug_file.close()


if __name__ == "__main__":
    print("Avvio del training...", flush=True)
    configure_training()
    train()
