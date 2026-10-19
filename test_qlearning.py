import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

import RL_tiago as env


# ============================================================
# CONFIGURAZIONE
# ============================================================

TEST_EPISODES = 100
MAX_STEPS = 500

# False: test senza animazione.
# True: mostra il robot e il pedone durante il test.
SHOW_ANIMATION = True

# Pausa tra i frame, se SHOW_ANIMATION=True.
DRAW_PAUSE = 0.2

# Seed per rendere riproducibile la sequenza casuale del pedone.
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CARICAMENTO CHECKPOINT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "q_table.pkl"
)

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Q-table non trovata: {CHECKPOINT_PATH}\n"
        "Esegui prima il training."
    )

with open(CHECKPOINT_PATH, "rb") as file:
    checkpoint = pickle.load(file)

q_values = checkpoint["q_values"]
task = checkpoint["task"]

# Le funzioni dell'ambiente utilizzano la variabile globale env.task.
env.task = task


# ============================================================
# VERIFICA COMPATIBILITÀ
# ============================================================

if checkpoint["rows"] != env.rows:
    raise ValueError(
        "Il numero di righe non corrisponde al training."
    )

if checkpoint["columns"] != env.columns:
    raise ValueError(
        "Il numero di colonne non corrisponde al training."
    )

if checkpoint["actions"] != env.actions:
    raise ValueError(
        "Le azioni di movimento non corrispondono al training."
    )

if checkpoint["communication_actions"] != env.communication_actions:
    raise ValueError(
        "Le azioni di comunicazione non corrispondono al training."
    )

if task not in (1, 2):
    raise ValueError(
        f"Task non valido nel checkpoint: {task}"
    )

if any(
    len(values) != env.n_q_actions
    for values in q_values.values()
):
    raise ValueError(
        "La Q-table contiene vettori con dimensioni incompatibili."
    )


print("\n========== CONFIGURAZIONE TEST ==========")
print(f"Q-table: {CHECKPOINT_PATH}")
print(f"Task: {task}")
print(f"Griglia: {env.rows} x {env.columns}")
print(f"Stati appresi: {len(q_values)}")
print(f"Episodi di test: {TEST_EPISODES}")
print(f"Massimo step: {MAX_STEPS}")
print(f"Animazione: {SHOW_ANIMATION}")


# ============================================================
# POSIZIONI INIZIALI
# ============================================================

print("\nLe coordinate partono da 0.")

start_row = int(
    input("Inserisci la riga iniziale del robot: ")
)

start_column = int(
    input("Inserisci la colonna iniziale del robot: ")
)

start_ped_row = int(
    input("Inserisci la riga iniziale del pedone: ")
)

start_ped_column = int(
    input("Inserisci la colonna iniziale del pedone: ")
)


def validate_positions():

    positions = [
        ("Robot", start_row, start_column),
        ("Pedone", start_ped_row, start_ped_column)
    ]

    for name, row, column in positions:

        if not (
            0 <= row < env.rows
            and 0 <= column < env.columns
        ):
            raise ValueError(
                f"Posizione iniziale del {name} fuori dalla griglia: "
                f"({row}, {column})"
            )

    if (
        start_row,
        start_column
    ) == (
        start_ped_row,
        start_ped_column
    ):
        raise ValueError(
            "Robot e pedone non possono partire dalla stessa cella."
        )


validate_positions()


# ============================================================
# VISUALIZZAZIONE
# ============================================================

if SHOW_ANIMATION:

    fig, ax1 = plt.subplots(figsize=(7, 7))

    plt.ion()

else:

    fig = None
    ax1 = None


def current_draw_state(
    robot_row,
    robot_column,
    ped_row,
    ped_column,
    episode,
    steps,
    robot_orientation,
    ped_orientation,
    robot_visibility,
    ped_visibility,
    communication_action
):

    if not SHOW_ANIMATION:
        return

    ax1.clear()

    # --------------------------------------------------------
    # GRIGLIA
    # --------------------------------------------------------

    ax1.set_xlim(-0.5, env.columns - 0.5)
    ax1.set_ylim(env.rows - 0.5, -0.5)

    ax1.set_xticks(range(env.columns))
    ax1.set_yticks(range(env.rows))

    ax1.grid(True)

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    if task == 1:

        coverage = env.coverage_matrix

        robot_rows, robot_columns = np.where(
            coverage == 1
        )

        ax1.scatter(
            robot_columns,
            robot_rows,
            color="cyan",
            s=180,
            label="Celle visitate dal robot"
        )

        ped_rows, ped_columns = np.where(
            coverage == -1
        )

        ax1.scatter(
            ped_columns,
            ped_rows,
            color="orange",
            s=180,
            label="Celle del pedone"
        )

    # --------------------------------------------------------
    # VISIBILITÀ
    # --------------------------------------------------------

    if robot_visibility and ped_visibility:

        robot_color = "green"
        ped_color = "green"

    elif robot_visibility:

        robot_color = "red"
        ped_color = "purple"

    else:

        robot_color = "blue"
        ped_color = "purple"

    markers = {
        "UP": "^",
        "DOWN": "v",
        "LEFT": "<",
        "RIGHT": ">"
    }

    # --------------------------------------------------------
    # ROBOT
    # --------------------------------------------------------

    ax1.scatter(
        robot_column,
        robot_row,
        color=robot_color,
        marker=markers.get(robot_orientation, "o"),
        s=300,
        edgecolors="black",
        label="Robot",
        zorder=4
    )

    # --------------------------------------------------------
    # PEDONE
    # --------------------------------------------------------

    ax1.scatter(
        ped_column,
        ped_row,
        color=ped_color,
        marker=markers.get(ped_orientation, "o"),
        s=300,
        edgecolors="black",
        label="Pedone",
        zorder=4
    )

    # --------------------------------------------------------
    # COMUNICAZIONE
    # --------------------------------------------------------

    communication_styles = {
        "none": {
            "color": "gray",
            "linestyle": "-",
            "label": "Nessuna"
        },
        "acoustic": {
            "color": "darkorange",
            "linestyle": "--",
            "label": "Acustica"
        },
        "visual": {
            "color": "blue",
            "linestyle": "--",
            "label": "Visiva"
        },
        "projection": {
            "color": "magenta",
            "linestyle": ":",
            "label": "Proiezione"
        }
    }

    communication = communication_styles.get(
        communication_action,
        communication_styles["none"]
    )

    # Linea tra robot e pedone per le modalità attive.
    if communication_action != "none":

        ax1.plot(
            [robot_column, ped_column],
            [robot_row, ped_row],
            color=communication["color"],
            linestyle=communication["linestyle"],
            linewidth=2.5,
            zorder=2
        )

    # Indicazione testuale sempre visibile.
    ax1.text(
        0.02,
        0.98,
        f"Comunicazione: {communication['label']}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(
            facecolor="white",
            edgecolor=communication["color"],
            alpha=0.9,
            pad=5
        ),
        zorder=5
    )

    ax1.set_title(
        f"Task {task} | Episodio {episode + 1} | Step {steps}"
    )

    ax1.set_xlabel("Colonne")
    ax1.set_ylabel("Righe")

    ax1.legend(loc="upper right", fontsize=8)

    plt.pause(DRAW_PAUSE)


# ============================================================
# SELEZIONE DELL'AZIONE
# ============================================================

unknown_state_decisions = 0


def get_action(
    state,
    robot_row,
    robot_column,
    robot_orientation,
    ped_row,
    ped_column
):

    global unknown_state_decisions

    legal_movements = env.get_legal_actions(
        robot_row,
        robot_column,
        robot_orientation,
        ped_row,
        ped_column
    )

    if not legal_movements:
        return None

    possible_actions = [
        env.encode_action(movement, communication)
        for movement in legal_movements
        for communication in range(
            len(env.communication_actions)
        )
    ]

    state_key = tuple(state)

    # get() evita di aggiungere stati alla Q-table.
    state_q_values = q_values.get(state_key)

    if state_q_values is None:

        unknown_state_decisions += 1

        # Fallback deterministico per uno stato sconosciuto.
        return int(possible_actions[0])

    legal_q_values = state_q_values[possible_actions]

    best_index = int(np.argmax(legal_q_values))

    return int(possible_actions[best_index])


# ============================================================
# TEST
# ============================================================

def test():

    global unknown_state_decisions

    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_communication_rewards = []

    total_communication_counts = {
        communication: 0
        for communication in env.communication_actions
    }

    unknown_state_decisions = 0

    # --------------------------------------------------------
    # COPIA DELLA Q-TABLE
    # --------------------------------------------------------

    q_table_size_before = len(q_values)

    q_values_before = {
        state: np.array(values, copy=True)
        for state, values in q_values.items()
    }

    # --------------------------------------------------------
    # EPISODI
    # --------------------------------------------------------

    for episode in range(TEST_EPISODES):

        if task == 2:
            env.reset_passing_side()

        print(
            f"\n========== EPISODIO "
            f"{episode + 1}/{TEST_EPISODES} =========="
        )

        robot_row = start_row
        robot_column = start_column

        ped_row = start_ped_row
        ped_column = start_ped_column

        robot_orientation = "UP"
        ped_orientation = "DOWN"

        episode_reward = 0.0
        communication_reward_total = 0.0

        steps = 0
        done = False

        episode_communication_counts = {
            communication: 0
            for communication in env.communication_actions
        }

        # ----------------------------------------------------
        # INIZIALIZZAZIONE TASK 1
        # ----------------------------------------------------

        if task == 1:

            env.coverage_matrix.fill(0)

            env.coverage_matrix[
                ped_row,
                ped_column
            ] = -1

            env.update_coverage_matrix(
                robot_row,
                robot_column
            )

        # ----------------------------------------------------
        # CICLO DEGLI STEP
        # ----------------------------------------------------

        while not done and steps < MAX_STEPS:

            steps += 1

            # ================================================
            # 1. OSSERVAZIONE
            # ================================================

            distance = env.calculate_distance(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            robot_visibility, ped_visibility = env.visibility(
                robot_orientation,
                ped_orientation,
                robot_row,
                robot_column,
                ped_row,
                ped_column
            )

            previous_robot_visibility = robot_visibility

            observed_ped_orientation = (
                ped_orientation
                if robot_visibility
                else None
            )

            state = (
                distance,
                robot_orientation,
                observed_ped_orientation,
                robot_visibility,
                ped_visibility
            )

            # ================================================
            # 2. SELEZIONE AZIONE GREEDY
            # ================================================

            action_index = get_action(
                state,
                robot_row,
                robot_column,
                robot_orientation,
                ped_row,
                ped_column
            )

            if action_index is None:

                print(
                    "Nessuna azione legale disponibile."
                )

                break

            movement_index, communication_index = (
                env.decode_action(action_index)
            )

            communication_action = (
                env.communication_actions[
                    communication_index
                ]
            )

            episode_communication_counts[
                communication_action
            ] += 1

            total_communication_counts[
                communication_action
            ] += 1

            # ================================================
            # 3. POSIZIONE PRECEDENTE
            # ================================================

            old_row = robot_row
            old_column = robot_column
            old_orientation = robot_orientation

            # ================================================
            # 4. MOVIMENTO ROBOT
            # ================================================

            (
                new_row,
                new_column,
                new_orientation,
                moved,
                collision
            ) = env.get_next_state(
                robot_row,
                robot_column,
                robot_orientation,
                ped_row,
                ped_column,
                movement_index
            )

            # ================================================
            # 5. REWARD
            # ================================================

            if collision:

                (
                    new_row,
                    new_column,
                    new_orientation,
                    reward,
                    _
                ) = env.use_shield(
                    old_row,
                    old_column,
                    old_orientation
                )

                reward_comm = 0.0

            else:

                (
                    new_robot_visibility,
                    new_ped_visibility
                ) = env.visibility(
                    new_orientation,
                    ped_orientation,
                    new_row,
                    new_column,
                    ped_row,
                    ped_column
                )

                new_distance = env.calculate_distance(
                    new_row,
                    new_column,
                    ped_row,
                    ped_column,
                    new_orientation
                )

                reward_task, done = env.get_task_reward(
                    old_row,
                    old_column,
                    new_row,
                    new_column,
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

                reward_comm = env.get_communication_reward(
                    communication_action,
                    new_robot_visibility,
                    new_ped_visibility,
                    new_distance
                )

                reward = reward_task + reward_comm

            # ================================================
            # 6. AGGIORNAMENTO ROBOT
            # ================================================

            robot_row = new_row
            robot_column = new_column
            robot_orientation = new_orientation

            # ================================================
            # 7. AGGIORNAMENTO COVERAGE
            # ================================================

            if task == 1:

                env.update_coverage_matrix(
                    robot_row,
                    robot_column
                )

            # ================================================
            # 8. CONTROLLO TASK 1
            # ================================================

            if task == 1:

                # Verifica la copertura prima di muovere il pedone.
                # Se il task è completato, non serve un altro movimento.
                done = env.is_coverage_completed()

                if done:

                    reward += 10

                else:

                    (
                        ped_row,
                        ped_column,
                        ped_orientation
                    ) = env.update_pedestrian_position(
                        ped_row,
                        ped_column,
                        robot_row,
                        robot_column,
                        ped_orientation
                    )

                    # Mantiene aggiornata la cella occupata dal pedone.
                    if env.coverage_matrix[
                        ped_row,
                        ped_column
                    ] == 0:

                        env.coverage_matrix[
                            ped_row,
                            ped_column
                        ] = -1

            # Nel task 2, il pedone si muove soltanto se il task
            # non è già terminato.
            elif not done:

                (
                    ped_row,
                    ped_column,
                    ped_orientation
                ) = env.update_pedestrian_position(
                    ped_row,
                    ped_column,
                    robot_row,
                    robot_column,
                    ped_orientation
                )

            # ================================================
            # 9. ACCUMULO REWARD
            # ================================================

            episode_reward += reward
            communication_reward_total += reward_comm

            # ================================================
            # 10. VISUALIZZAZIONE
            # ================================================

            current_draw_state(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                episode,
                steps,
                robot_orientation,
                ped_orientation,
                robot_visibility,
                ped_visibility,
                communication_action
            )

            print(
                f"Step {steps:03d} | "
                f"Robot=({robot_row},{robot_column}) | "
                f"Pedone=({ped_row},{ped_column}) | "
                f"Movimento={env.actions[movement_index]} | "
                f"Comunicazione={communication_action} | "
                f"Reward={reward:.2f} | "
                f"Totale={episode_reward:.2f}"
            )

        # ----------------------------------------------------
        # FINE EPISODIO
        # ----------------------------------------------------

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_success.append(done)

        episode_communication_rewards.append(
            communication_reward_total
        )

        status = (
            "SUCCESSO"
            if done
            else "NON COMPLETATO"
        )

        print(
            f"\nEpisodio {episode + 1}: "
            f"{status} | "
            f"Reward={episode_reward:.2f} | "
            f"Step={steps}"
        )

        print(
            "Comunicazioni:",
            episode_communication_counts
        )

    # ========================================================
    # STATISTICHE FINALI
    # ========================================================

    successful_steps = [
        steps
        for steps, success in zip(
            episode_steps,
            episode_success
        )
        if success
    ]

    failed_steps = [
        steps
        for steps, success in zip(
            episode_steps,
            episode_success
        )
        if not success
    ]

    success_rate = (
        100 * np.mean(episode_success)
        if episode_success
        else 0.0
    )

    print("\n========== RISULTATI FINALI ==========")

    print(f"Task: {task}")
    print(f"Episodi: {TEST_EPISODES}")
    print(f"Episodi completati: {sum(episode_success)}")
    print(f"Episodi non completati: {len(episode_success) - sum(episode_success)}")
    print(f"Tasso di successo: {success_rate:.2f}%")

    print(
        f"Reward medio: "
        f"{np.mean(episode_rewards):.2f}"
    )

    print(
        f"Reward comunicativo medio: "
        f"{np.mean(episode_communication_rewards):.2f}"
    )

    print(
        f"Step medi complessivi: "
        f"{np.mean(episode_steps):.2f}"
    )

    print(
        "Step medi negli episodi completati: "
        + (
            f"{np.mean(successful_steps):.2f}"
            if successful_steps
            else "Nessun episodio completato"
        )
    )

    print(
        "Step medi negli episodi non completati: "
        + (
            f"{np.mean(failed_steps):.2f}"
            if failed_steps
            else "Nessun episodio fallito"
        )
    )

    print(
        f"Decisioni prese in stati sconosciuti: "
        f"{unknown_state_decisions}"
    )

    # --------------------------------------------------------
    # STATISTICHE COMUNICAZIONE
    # --------------------------------------------------------

    total_communications = sum(
        total_communication_counts.values()
    )

    print("\nComunicazioni complessive:")

    for communication, count in total_communication_counts.items():

        percentage = (
            100 * count / total_communications
            if total_communications > 0
            else 0.0
        )

        print(
            f"  {communication}: {count} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # VERIFICA IMMUTABILITÀ Q-TABLE
    # --------------------------------------------------------

    assert len(q_values) == q_table_size_before, (
        "Il numero di stati della Q-table è cambiato."
    )

    assert all(
        np.array_equal(
            q_values[state],
            old_values
        )
        for state, old_values in q_values_before.items()
    ), (
        "I Q-value sono stati modificati durante il test."
    )

    print("\nOK: la Q-table è rimasta invariata.")

    if SHOW_ANIMATION:
        plt.ioff()
        plt.show()


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    test()
