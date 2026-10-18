passing_phase = 0
passing_direction = None
passing_sequence = []

# Fasi:
# 0 = attesa dell'avvio del task
# 1 = manovra in corso
# 2 = task completato

SUCCESS_REWARD = 10.0
START_REWARD = 0.5
STEP_REWARD = 0.2
INVALID_REWARD = -0.1

TARGET_DISTANCE = 1


def reset_passing_side():
    global passing_phase, passing_direction, passing_sequence

    passing_phase = 0
    passing_direction = None
    passing_sequence = []


def abort_passing_side():
    reset_passing_side()


def reward_passing_side(
    previous_robot_observation,
    robot_observation,
    distance,
    turn_left_result,
    turn_right_result,
    go_forward_result,
    executed_action
):
    global passing_phase, passing_direction, passing_sequence

    # ========================================================
    # PHASE 0: avvio del task
    # ========================================================
    if passing_phase == 0:

        # Il pedone deve essere stato osservato prima dell'azione.
        if not previous_robot_observation:
            return INVALID_REWARD, False

        # Il robot sceglie il lato del passaggio.
        if executed_action == "left_forward":

            if not turn_left_result or not go_forward_result:
                return INVALID_REWARD, False

            passing_direction = "left"

        elif executed_action == "right_forward":

            if not turn_right_result or not go_forward_result:
                return INVALID_REWARD, False

            passing_direction = "right"

        else:
            return INVALID_REWARD, False

        passing_sequence = [executed_action]
        passing_phase = 1

        return START_REWARD, False

    # ========================================================
    # PHASE 1: manovra in corso
    # ========================================================
    elif passing_phase == 1:

        # Il robot può avanzare o ruotare durante la manovra.
        valid_actions = (
            "forward",
            "left_forward",
            "right_forward",
            "turn_left",
            "turn_right"
        )

        if executed_action not in valid_actions:
            abort_passing_side()
            return INVALID_REWARD, False

        # Verifica che l'azione sia stata eseguita correttamente.
        if executed_action == "forward" and not go_forward_result:
            return STEP_REWARD, False

        if executed_action == "turn_left" and not turn_left_result:
            return STEP_REWARD, False

        if executed_action == "turn_right" and not turn_right_result:
            return STEP_REWARD, False

        if executed_action == "left_forward":
            if not turn_left_result or not go_forward_result:
                return STEP_REWARD, False

        if executed_action == "right_forward":
            if not turn_right_result or not go_forward_result:
                return STEP_REWARD, False

        passing_sequence.append(executed_action)

        # SUCCESSO:
        # il pedone è visibile e la distanza è quella desiderata.
        if (
            robot_observation
            and distance == TARGET_DISTANCE
        ):
            passing_phase = 2
            return SUCCESS_REWARD, True

        # Il task continua.
        return STEP_REWARD, False

    # ========================================================
    # PHASE 2: task completato
    # ========================================================
    elif passing_phase == 2:
        return 0.0, True

    abort_passing_side()
    return INVALID_REWARD, False