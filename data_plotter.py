import matplotlib.pyplot as plt
from stable_baselines import results_plotter

TIME_STEPS = 500000

models = ["dqn", "a2c", "ppo2"]

log_dir = "./{}_logs/".format()
results_plotter.plot_results([log_dir], TIME_STEPS, results_plotter.X_TIMESTEPS, "Rewards over episodes")
plt.show()