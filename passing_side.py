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
