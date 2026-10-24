import os
import pickle
import random
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import RL_tiago_debug as env
import coverage as cov


# ============================================================
# CONFIGURAZIONE
# ============================================================

TEST_EPISODES = 100
MAX_STEPS = 500

SHOW_ANIMATION = False
DRAW_PAUSE = 0.001
RANDOM_SEED = 42

# Se True, scrive nel TXT anche tutti i dettagli di ogni step.
# Se False, registra solo configurazione, riepiloghi e anomalie.
DEBUG_EVERY_STEP = True

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CARICAMENTO CHECKPOINT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "q_table.pkl")
DEBUG_PATH = os.path.join(
    BASE_DIR,
    f"debug_test_qlearning_{datetime.now():%Y%m%d_%H%M%S}.txt"
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
env.task = task


# ============================================================
# VERIFICA COMPATIBILITÀ
# ============================================================

if checkpoint["rows"] != env.rows:
    raise ValueError("Il numero di righe non corrisponde al training.")

if checkpoint["columns"] != env.columns:
    raise ValueError("Il numero di colonne non corrisponde al training.")

if checkpoint["actions"] != env.actions:
    raise ValueError("Le azioni di movimento non corrispondono al training.")

if len(env.actions) != env.n_q_actions:
    raise ValueError(
        f"Configurazione incoerente: env.actions contiene "
        f"{len(env.actions)} azioni, mentre env.n_q_actions vale "
        f"{env.n_q_actions}."
    )

if task not in (1, 2):
    raise ValueError(f"Task non valido nel checkpoint: {task}")

if any(len(values) != env.n_q_actions for values in q_values.values()):
    raise ValueError("La Q-table contiene vettori con dimensioni incompatibili.")


print("\n========== CONFIGURAZIONE TEST ==========")
print(f"Q-table: {CHECKPOINT_PATH}")
print(f"Task: {task}")
print(f"Griglia: {env.rows} x {env.columns}")
print(f"Stati appresi: {len(q_values)}")
print(f"Episodi di test: {TEST_EPISODES}")
print(f"Massimo step: {MAX_STEPS}")
print(f"Animazione: {SHOW_ANIMATION}")
print(f"Debug TXT: {DEBUG_PATH}")


# ============================================================
# POSIZIONI INIZIALI
# ============================================================

print("\nLe coordinate partono da 0.")

start_row = int(input("Inserisci la riga iniziale del robot: "))
start_column = int(input("Inserisci la colonna iniziale del robot: "))
start_ped_row = int(input("Inserisci la riga iniziale del pedone: "))
start_ped_column = int(input("Inserisci la colonna iniziale del pedone: "))


def validate_positions():
    positions = [
        ("Robot", start_row, start_column),
        ("Pedone", start_ped_row, start_ped_column)
    ]

    for name, row, column in positions:
        if not (0 <= row < env.rows and 0 <= column < env.columns):
            raise ValueError(
                f"Posizione iniziale del {name} fuori dalla griglia: "
                f"({row}, {column})"
            )

    if (start_row, start_column) == (start_ped_row, start_ped_column):
        raise ValueError("Robot e pedone non possono partire dalla stessa cella.")


validate_positions()


# ============================================================
# LOGGING TXT
# ============================================================

logger = logging.getLogger("qlearning_test")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

_file_handler = logging.FileHandler(DEBUG_PATH, mode="w", encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S")
)
logger.addHandler(_file_handler)


def log_info(message):
    logger.info(message)


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
    robot_row, robot_column, ped_row, ped_column, episode, steps,
    robot_orientation, ped_orientation, robot_visibility, ped_visibility,
    action
):
    if not SHOW_ANIMATION:
        return

    ax1.clear()
    ax1.set_xlim(-0.5, env.columns - 0.5)
    ax1.set_ylim(env.rows - 0.5, -0.5)
    ax1.set_xticks(range(env.columns))
    ax1.set_yticks(range(env.rows))
    ax1.grid(True)

    # ============================================================
    # CELLE VISITATE
    # Visualizzale soltanto quando il robot osserva il pedone.
    # ============================================================

    if task == 1:
        coverage = cov.coverage_matrix

        robot_rows, robot_columns = np.where(coverage == 1)
        ped_rows, ped_columns = np.where(coverage == -1)

        ax1.scatter(
            robot_columns,
            robot_rows,
            color="cyan",
            s=180,
            label="Celle visitate dal robot",
            zorder=1
        )

        ax1.scatter(
            ped_columns,
            ped_rows,
            color="orange",
            s=180,
            label="Celle visitate dal pedone",
            zorder=1
        )

    # ============================================================
    # ROBOT
    # ============================================================

    if robot_visibility and ped_visibility:
        robot_color = "green"
    elif robot_visibility:
        robot_color = "red"
    else:
        robot_color = "blue"

    markers = {
        "UP": "^",
        "DOWN": "v",
        "LEFT": "<",
        "RIGHT": ">"
    }

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

    # ============================================================
    # PEDONE
    # Sempre visibile nell'animazione, indipendentemente dalla
    # visibilità percepita dal robot.
    # ============================================================

    ax1.scatter(
        ped_column,
        ped_row,
        color="purple",
        marker=markers.get(ped_orientation, "o"),
        s=300,
        edgecolors="black",
        label="Pedone",
        zorder=4
    )

    # ============================================================
    # COMUNICAZIONE
    # ============================================================

    communication_styles = {
        3: {
            "color": "darkorange",
            "linestyle": "--",
            "label": "Acustica"
        },
        4: {
            "color": "blue",
            "linestyle": "--",
            "label": "Visiva"
        },
        5: {
            "color": "magenta",
            "linestyle": ":",
            "label": "Proiezione"
        }
    }

    communication = communication_styles.get(action)

    if communication is not None:
        if robot_visibility:
            ax1.plot(
                [robot_column, ped_column],
                [robot_row, ped_row],
                color=communication["color"],
                linestyle=communication["linestyle"],
                linewidth=2.5,
                zorder=2
            )

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
        f"Episodio {episode + 1} | Step {steps} | "
        f"Azione: {env.actions[action] if action is not None else 'Nessuna'}"
    )

    ax1.legend(loc="upper right")
    fig.canvas.draw_idle()
    plt.pause(DRAW_PAUSE)


# ============================================================
# SELEZIONE DELL'AZIONE
# ============================================================

unknown_state_decisions = 0


def get_action(state, robot_row, robot_column, robot_orientation):
    global unknown_state_decisions

    # Ottieni le azioni legali secondo i vincoli dell'ambiente.
    legal_actions = env.get_legal_actions(
        robot_row,
        robot_column,
        robot_orientation
    )

    if not legal_actions:
        return None, [], False

    state_key = tuple(state)
    state_q_values = q_values.get(state_key)

    if state_q_values is None:
        unknown_state_decisions += 1
        return random.choice(legal_actions), legal_actions, True

    # Valuta soltanto le azioni di movimento legali.
    legal_q_values = state_q_values[legal_actions]
    max_q = np.max(legal_q_values)

    # In caso di parità, scegli casualmente tra le migliori.
    best_actions = [
        action for action in legal_actions
        if state_q_values[action] == max_q
    ]

    return random.choice(best_actions), legal_actions, False


# ============================================================
# TEST
# ============================================================

def test():
    global unknown_state_decisions

    episode_rewards = []
    episode_steps = []
    episode_success = []
    episode_collision_counts = []
    episode_shield_counts = []
    episode_unknown_counts = []
    episode_no_move_counts = []
    episode_summaries = []

    communication_names = {
        3: "acoustic",
        4: "visual",
        5: "projection"
    }

    total_communication_counts = {
        name: 0 for name in communication_names.values()
    }

    unknown_state_decisions = 0
    q_table_size_before = len(q_values)
    q_values_before = {
        state: np.array(values, copy=True) for state, values in q_values.items()
    }

    log_info("=" * 72)
    log_info("INIZIO TEST Q-LEARNING")
    log_info(f"Checkpoint={CHECKPOINT_PATH}")
    log_info(f"Task={task}; griglia={env.rows}x{env.columns}")
    log_info(f"Episodi={TEST_EPISODES}; max_steps={MAX_STEPS}")
    log_info(f"Seed={RANDOM_SEED}; animazione={SHOW_ANIMATION}")
    log_info(f"Stati Q-table={len(q_values)}")
    log_info(
        f"Posizioni iniziali robot=({start_row},{start_column}), "
        f"pedone=({start_ped_row},{start_ped_column})"
    )
    log_info(f"DEBUG_EVERY_STEP={DEBUG_EVERY_STEP}")

    for episode in range(TEST_EPISODES):
        episode_communication_counts = {
            name: 0 for name in communication_names.values()
        }
        if task == 2:
            env.reset_passing_side()

        print(f"\n========== EPISODIO {episode + 1}/{TEST_EPISODES} ==========")
        log_info("-" * 72)
        log_info(f"EPISODIO {episode + 1} START")

        robot_row, robot_column = start_row, start_column
        ped_row, ped_column = start_ped_row, start_ped_column
        robot_orientation = "UP"
        ped_orientation = "DOWN"
        awareness = 0.0

        episode_reward = 0.0
        communication_reward_total = 0.0
        steps = 0
        done = False
        termination_reason = "max_steps"
        collision_count = 0
        shield_count = 0
        unknown_count_start = unknown_state_decisions
        no_move_count = 0

        if task == 1:
            cov.coverage_matrix.fill(0)

            # Il training registra inizialmente la posizione del pedone
            # come -1 e quella del robot come 1.
            cov.update_coverage_matrix(ped_row, ped_column, -1)

            cov.update_coverage_matrix(robot_row, robot_column, 1)

        while not done and steps < MAX_STEPS:
            steps += 1

            # ====================================================
            # 1. OSSERVAZIONE DELLO STATO CORRENTE
            # ====================================================

            robot_visibility = env.get_robot_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            ped_visibility = env.get_pedestrian_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                ped_orientation
            )

            if robot_visibility:
                distance = env.calculate_distance(
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

            previous_robot_visibility = robot_visibility
            previous_distance = distance

            # 2. Costruzione dello stato: stessi argomenti del training.
            state = env.build_state(
                distance,
                robot_orientation,
                observed_ped_orientation,
                robot_visibility,
                ped_visibility,
                awareness
            )

            # ====================================================
            # 2. SELEZIONE DELL'AZIONE GREEDY
            # ====================================================

            action_index, possible_actions, unknown_state = get_action(
                state,
                robot_row,
                robot_column,
                robot_orientation
            )

            if action_index is not None:
                awareness = env.get_awareness(
                    ped_visibility,
                    robot_visibility,
                    action_index,
                    awareness
                )

            if task == 1:
                if robot_visibility:
                    ped_row_visibile = ped_row
                    ped_column_visibile = ped_column

                    cov.update_coverage_matrix(ped_row_visibile, ped_column_visibile, -1)

            if unknown_state:
                log_info(
                    f"EP={episode + 1} STEP={steps} | "
                    f"STATO NON PRESENTE NELLA Q-TABLE | "
                    f"state={tuple(state)!r} | "
                    f"azione selezionata casualmente tra quelle legali"
                )

                print(
                    "Stato sconosciuto: selezionata un'azione legale casuale."
                )

            if action_index is None:
                termination_reason = "no_legal_actions"

                log_info(
                    f"EP={episode + 1} STEP={steps} | "
                    "NESSUNA AZIONE LEGALE"
                )
                break

            # ====================================================
            # DIAGNOSTICA DEI VALORI Q
            # ====================================================

            state_q_values = q_values.get(tuple(state))

            if state_q_values is None:
                # Stato sconosciuto: l'azione è stata scelta casualmente.
                q_diagnostic = {
                    "q_scelta": None,
                    "q_stesso_movimento": None,
                    "tutti_q_zero": None,
                    "stato_sconosciuto": True
                }

            else:
                q_diagnostic = {
                    "q_scelta": round(
                        float(state_q_values[action_index]), 4
                    ),
                    "tutti_q_zero": bool(
                        np.all(state_q_values == 0)
                    ),
                    "stato_sconosciuto": False
                }

            # ====================================================
            # 3. ESECUZIONE DEL MOVIMENTO DEL ROBOT
            # ====================================================

            old_row = robot_row
            old_column = robot_column
            old_orientation = robot_orientation

            (
                candidate_row,
                candidate_column,
                candidate_orientation,
                candidate_moved
            ) = env.get_next_position(
                old_row,
                old_column,
                old_orientation,
                action_index
            )

            collision = (
                action_index in (0, 1, 2)
                and candidate_moved
                and (candidate_row, candidate_column)
                == (ped_row, ped_column)
            )

            # ====================================================
            # 4. GESTIONE DELLA TRANSIZIONE
            # ====================================================

            done = False
            reward = 0.0
            new_distance = distance

            if collision:
                # Stessa penalità prevista dal training.
                collision_count += 1
                shield_count += 1

                robot_row = old_row
                robot_column = old_column
                robot_orientation = old_orientation

                moved = False
                reward = -10.0

            else:
                # Applica la transizione candidata.
                robot_row = candidate_row
                robot_column = candidate_column
                robot_orientation = candidate_orientation
                moved = candidate_moved

                # Calcola le nuove osservazioni.
                robot_visibility = env.get_robot_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    robot_orientation
                )

                ped_visibility = env.get_pedestrian_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    ped_orientation
                )

                if robot_visibility:
                    new_distance = env.calculate_distance(
                        robot_row,
                        robot_column,
                        ped_row,
                        ped_column,
                        robot_orientation
                    )
                else:
                    new_distance = float("inf")

                # Usa la funzione di ricompensa del training.
                reward, done = env.get_reward(
                    old_row,
                    old_column,
                    robot_row,
                    robot_column,
                    previous_distance,
                    new_distance,
                    action_index,
                    previous_robot_visibility,
                    robot_visibility,
                    robot_orientation,
                    moved
                )

            if not moved:
                no_move_count += 1

            # ====================================================
            # 6. AGGIORNAMENTO DELLO STATO DEL ROBOT
            # ====================================================

            if task == 1:
                cov.update_coverage_matrix(
                    robot_row,
                    robot_column,
                    1
                )

                if robot_visibility:
                    cov.update_coverage_matrix(
                        ped_row,
                        ped_column,
                        -1
                    )

            old_robot_row = robot_row
            old_robot_column = robot_column
            old_robot_orientation = robot_orientation

            # ====================================================
            # 6.1 VISUALIZZA IL MOVIMENTO DEL ROBOT
            # Il pedone mantiene ancora la posizione precedente.
            # ====================================================

            # Calcola le nuove osservazioni.
            robot_visibility_after_move = env.get_robot_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            ped_visibility_before_move = env.get_pedestrian_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                ped_orientation
            )

            current_draw_state(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                episode,
                steps,
                robot_orientation,
                ped_orientation,
                robot_visibility_after_move,
                ped_visibility_before_move,
                action_index
            )

            ped_row, ped_column, ped_orientation, moved, old_robot_row, old_robot_column = (
                env.update_pedestrian_position(
                    ped_row,
                    ped_column,
                    old_robot_row,
                    old_robot_column,
                    robot_row,
                    robot_column,
                    ped_orientation,
                    awareness
                )
            )

            # ====================================================
            # 7.1 VISUALIZZA IL MOVIMENTO DEL PEDONE
            # Il robot rimane fermo nella posizione raggiunta.
            # ====================================================

            # Calcola le nuove osservazioni.
            robot_visibility_after_ped_move = env.get_robot_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                robot_orientation
            )

            ped_visibility_after_move = env.get_pedestrian_visibility(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                ped_orientation
            )

            current_draw_state(
                robot_row,
                robot_column,
                ped_row,
                ped_column,
                episode,
                steps,
                robot_orientation,
                ped_orientation,
                robot_visibility_after_ped_move,
                ped_visibility_after_move,
                action_index
            )

            if task == 1:
                next_robot_visibility = env.get_robot_visibility(
                    robot_row,
                    robot_column,
                    ped_row,
                    ped_column,
                    robot_orientation
                )

                if next_robot_visibility:
                    cov.update_coverage_matrix(
                        ped_row,
                        ped_column,
                        -1
                    )
            # ====================================================
            # 8. VERIFICA DEL COMPLETAMENTO DEL TASK
            # ====================================================

            if task == 1:
                # Verifica il completamento della copertura.
                done = cov.is_coverage_completed()

                # Diagnostica della copertura.
                total_cells = cov.coverage_matrix.size
                covered_cells = int(np.count_nonzero(
                    cov.coverage_matrix == 1
                ))
                remaining_cells = int(np.count_nonzero(
                    cov.coverage_matrix == 0
                ))

                log_info(
                    f"EP={episode + 1} STEP={steps} | "
                    f"COVERAGE_CHECK | "
                    f"covered={covered_cells}/{total_cells} | "
                    f"remaining={remaining_cells} | "
                    f"done={done}"
                )

                if done:
                    success = True
                    if task == 1:
                        termination_reason = "coverage_completed"
                    else:
                        termination_reason = "task_terminated"
                else:
                    success = False
                    termination_reason = "max_steps"

            elif task == 2 and done:
                termination_reason = "task_terminated"

            # ====================================================
            # 9. AGGIORNAMENTO DELLE STATISTICHE DELL'EPISODIO
            # ====================================================

            episode_reward += reward

            # ====================================================
            # 11. LOG DELLA TRANSIZIONE
            # ====================================================

            step_message = (
                f"EP={episode + 1} STEP={steps} | "
                f"state={state!r} | "
                f"legal_actions={possible_actions} | "
                f"action_index={action_index} | "
                f"azione={env.actions[action_index]} | "
                f"q_diagnostic={q_diagnostic} | "
                f"robot_before=({old_row},{old_column},"
                f"{old_orientation}) | "
                f"robot_after=({robot_row},{robot_column},"
                f"{robot_orientation}) | "
                f"ped_after=({ped_row},{ped_column},"
                f"{ped_orientation}) | "
                f"visibility_before=(robot:{previous_robot_visibility},"
                f"ped:{ped_visibility}) | "
                f"visibility_after_robot_move=(robot:{robot_visibility},"
                f"distance_before={distance!r} | "
                f"distance_after_robot_move={new_distance!r} | "
                f"moved={moved} | collision={collision} | "
                f"reward={reward:.3f} | "
                f"reward_step={reward:.3f} | "
                f"reward_total={episode_reward:.3f} | "
                f"done={done}"
            )

            if DEBUG_EVERY_STEP:
                log_info(step_message)
            elif collision or not moved or done:
                log_info("ANOMALIA/TERMINAZIONE | " + step_message)

            print(
                f"Step {steps:03d} | "
                f"Robot=({robot_row},{robot_column}) | "
                f"Pedone=({ped_row},{ped_column}) | "
                f"Movimento={env.actions[action_index]} | "
                f"Reward={reward:.2f} | "
                f"Totale={episode_reward:.2f}"
            )

        # if not done and termination_reason == "max_steps" and steps >= MAX_STEPS:
        #     termination_reason = "max_steps"
        # elif not done and termination_reason == "max_steps":
        #     termination_reason = "interrupted"

        episode_unknown = unknown_state_decisions - unknown_count_start
        if task == 1:
            success = (
                done
                and termination_reason == "coverage_completed"
            )
        else:
            success = (
                done
                and termination_reason == "task_terminated"
            )

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_success.append(success)
        episode_collision_counts.append(collision_count)
        episode_shield_counts.append(shield_count)
        episode_unknown_counts.append(episode_unknown)
        episode_no_move_counts.append(no_move_count)

        summary = {
            "episode": episode + 1,
            "success": success,
            "termination_reason": termination_reason,
            "reward": episode_reward,
            "steps": steps,
            "communication_reward": communication_reward_total,
            "collisions": collision_count,
            "shield_uses": shield_count,
            "unknown_states": episode_unknown,
            "no_move_steps": no_move_count,
            "communication_counts": dict(episode_communication_counts)
        }
        episode_summaries.append(summary)

        status = "SUCCESSO" if success else "NON COMPLETATO"
        print(
            f"\nEpisodio {episode + 1}: {status} | "
            f"Reward={episode_reward:.2f} | Step={steps} | "
            f"Motivo={termination_reason}"
        )

        log_info(
            f"EPISODIO {episode + 1} END | status={status} | "
            f"reason={termination_reason} | reward={episode_reward:.3f} | "
            f"steps={steps} | comm_reward={communication_reward_total:.3f} | "
            f"collisions={collision_count} | shield_uses={shield_count} | "
            f"unknown_states={episode_unknown} | no_move_steps={no_move_count} | "
            f"communications={episode_communication_counts} | "
        )

        log_info(
            f"EPISODIO {episode + 1} COMMUNICATION_STATS | "
        )
        print("Comunicazioni:", episode_communication_counts)

    # ========================================================
    # STATISTICHE FINALI
    # ========================================================

    successful_steps = [
        n_steps for n_steps, success in zip(episode_steps, episode_success) if success
    ]
    failed_steps = [
        n_steps for n_steps, success in zip(episode_steps, episode_success) if not success
    ]

    success_rate = 100 * np.mean(episode_success) if episode_success else 0.0
    total_communications = sum(total_communication_counts.values())
    total_collisions = sum(episode_collision_counts)
    total_shield_uses = sum(episode_shield_counts)
    total_no_move_steps = sum(episode_no_move_counts)
    total_unknown_decisions = sum(episode_unknown_counts)
    total_steps = sum(episode_steps)
    # ========================================================
    # UTILIZZO DELLA Q-TABLE
    # ========================================================

    # Numero di decisioni che hanno utilizzato stati non presenti
    # nella Q-table.
    total_unknown_decisions = sum(episode_unknown_counts)

    # Numero totale di decisioni effettuate.
    total_steps = sum(episode_steps)

    # Decisioni effettuate in stati presenti nella Q-table.
    known_state_decisions = total_steps - total_unknown_decisions

    # Percentuale di decisioni in stati conosciuti.
    known_state_rate = (
        100.0 * known_state_decisions / total_steps
        if total_steps > 0 else 0.0
    )

    # Percentuale di decisioni in stati sconosciuti.
    unknown_state_rate = (
        100.0 * total_unknown_decisions / total_steps
        if total_steps > 0 else 0.0
    )

    print("\n========== RISULTATI FINALI ==========")
    print(f"Task: {task}")
    print(f"Episodi: {TEST_EPISODES}")
    print(f"Episodi completati: {sum(episode_success)}")
    print(f"Episodi non completati: {len(episode_success) - sum(episode_success)}")
    print(f"Tasso di successo: {success_rate:.2f}%")
    print(f"Reward medio: {np.mean(episode_rewards):.2f}")
    print(f"Step medi complessivi: {np.mean(episode_steps):.2f}")
    print(
        "Step medi negli episodi completati: "
        + (f"{np.mean(successful_steps):.2f}" if successful_steps else "Nessun episodio completato")
    )
    print(
        "Step medi negli episodi non completati: "
        + (f"{np.mean(failed_steps):.2f}" if failed_steps else "Nessun episodio fallito")
    )
    print(f"Decisioni in stati sconosciuti: {total_unknown_decisions}")
    print(f"Collisioni / attivazioni shield: {total_collisions} / {total_shield_uses}")
    print(f"Step senza movimento: {total_no_move_steps}")
    print("\nUtilizzo della Q-table:")
    print(f"Decisioni totali: {total_steps}")
    print(f"Decisioni in stati conosciuti: {known_state_decisions}")
    print(f"Decisioni in stati sconosciuti: {total_unknown_decisions}")
    print(f"Percentuale stati conosciuti: {known_state_rate:.2f}%")
    print(f"Percentuale stati sconosciuti: {unknown_state_rate:.2f}%")

    print("\nComunicazioni complessive:")
    for communication, count in total_communication_counts.items():
        percentage = 100 * count / total_communications if total_communications else 0.0
        print(f"  {communication}: {count} ({percentage:.2f}%)")

    print("\nReward per tipo di comunicazione:")

    log_info("=" * 72)
    log_info("RIEPILOGO FINALE")
    log_info(f"episodi={TEST_EPISODES}")
    log_info(f"successi={sum(episode_success)}")
    log_info(f"fallimenti={len(episode_success) - sum(episode_success)}")
    log_info(f"success_rate_percent={success_rate:.2f}")
    log_info(f"reward_mean={np.mean(episode_rewards):.4f}")
    log_info(f"reward_median={np.median(episode_rewards):.4f}")
    log_info(f"reward_min={np.min(episode_rewards):.4f}")
    log_info(f"reward_max={np.max(episode_rewards):.4f}")
    log_info(f"steps_mean={np.mean(episode_steps):.4f}")
    log_info(f"steps_median={np.median(episode_steps):.4f}")
    log_info(f"steps_success_mean={np.mean(successful_steps) if successful_steps else 'N/A'}")
    log_info(f"steps_failure_mean={np.mean(failed_steps) if failed_steps else 'N/A'}")
    log_info(f"collisions_total={total_collisions}")
    log_info(f"shield_uses_total={total_shield_uses}")
    log_info(f"no_move_steps_total={total_no_move_steps}")
    log_info(f"unknown_state_decisions_total={total_unknown_decisions}")
    log_info(f"total_steps={total_steps}")
    log_info(f"known_state_decisions_total={known_state_decisions}")
    log_info(f"known_state_rate_percent={known_state_rate:.2f}")
    log_info(f"unknown_state_rate_percent={unknown_state_rate:.2f}")
    log_info(f"communication_counts={total_communication_counts}")

    # Riepilogo per singolo episodio in un formato facilmente leggibile.
    log_info("RIEPILOGO PER EPISODIO (una riga per episodio)")
    for item in episode_summaries:
        log_info(
            "EPISODE_SUMMARY | "
            f"episode={item['episode']} | success={item['success']} | "
            f"reason={item['termination_reason']} | reward={item['reward']:.3f} | "
            f"steps={item['steps']} | comm_reward={item['communication_reward']:.3f} | "
            f"collisions={item['collisions']} | shield_uses={item['shield_uses']} | "
            f"unknown_states={item['unknown_states']} | "
            f"no_move_steps={item['no_move_steps']} | "
            f"communications={item['communication_counts']} | "
        )

    # ========================================================
    # VERIFICA IMMUTABILITÀ Q-TABLE
    # ========================================================

    q_table_unchanged = (
        len(q_values) == q_table_size_before
        and all(
            np.array_equal(q_values[state], old_values)
            for state, old_values in q_values_before.items()
        )
    )

    if not q_table_unchanged:
        log_info("ERRORE: la Q-table è stata modificata durante il test.")
        raise AssertionError("La Q-table è stata modificata durante il test.")

    print("\nOK: la Q-table è rimasta invariata.")
    log_info("VERIFICA Q-TABLE: OK, la Q-table è rimasta invariata.")
    log_info(f"FILE DEBUG: {DEBUG_PATH}")
    log_info("FINE TEST")

    if SHOW_ANIMATION:
        plt.ioff()
        plt.show()

    _file_handler.flush()
    _file_handler.close()
    logger.removeHandler(_file_handler)


# ============================================================
# AVVIO
# ============================================================

if __name__ == "__main__":
    try:
        test()
    except Exception:
        logger.exception("TEST INTERROTTO DA UN'ECCEZIONE")
        raise
    finally:
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
