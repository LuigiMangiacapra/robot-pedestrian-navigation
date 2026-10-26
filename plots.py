import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return np.array([])

    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_training_metrics():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "training_metrics.npz")

    data = np.load(metrics_path)

    episode_rewards = data["episode_rewards"]
    episode_success = data["episode_success"].astype(float)
    episode_steps = data["episode_steps"]
    episode_td_errors = data["episode_td_errors"]

    window = 100

    reward_avg = moving_average(episode_rewards, window)
    success_avg = moving_average(episode_success, window)
    steps_avg = moving_average(episode_steps, window)
    td_avg = moving_average(episode_td_errors, window)

    if len(reward_avg) == 0:
        print("Non ci sono abbastanza episodi per creare i grafici.")
        return

    episodes_x = np.arange(window, len(episode_rewards) + 1)

    # 1. Reward media mobile
    plt.figure()
    plt.plot(episodes_x, reward_avg)
    plt.xlabel("Episodio")
    plt.ylabel("Reward media")
    plt.title("Reward media mobile")
    plt.grid(True)
    plt.tight_layout()

    # 2. Percentuale di successo
    plt.figure()
    plt.plot(episodes_x, success_avg * 100)
    plt.xlabel("Episodio")
    plt.ylabel("Successi (%)")
    plt.title("Percentuale di successo mobile")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()

    # 3. Numero medio di passi
    plt.figure()
    plt.plot(episodes_x, steps_avg)
    plt.xlabel("Episodio")
    plt.ylabel("Passi medi")
    plt.title("Numero medio di passi")
    plt.grid(True)
    plt.tight_layout()

    # 4. Errore TD assoluto medio
    plt.figure()
    plt.plot(episodes_x, td_avg)
    plt.xlabel("Episodio")
    plt.ylabel("Errore TD assoluto medio")
    plt.title("Errore TD medio mobile")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_training_metrics()
