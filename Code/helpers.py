import numpy as np
from matplotlib import pyplot as plt


def plot_results_curves(data, block=True, title=None):

    bsf_list, avg_list, trajectory, v_history, w_history, a_history, f_history = data

    fig = plt.figure(3, figsize=(9, 7))
    title = 'BSF/AVG/Trajectory' if title is None else title
    fig.suptitle(title)

    # Disable lograitmic plot when there is negitive data
    log_plot = False if any(np.array(bsf_list) < 0) else True
    #  log_plot = False

    #  Best so far costs changes
    ax_bsf = plt.subplot(2, 2, 1)
    ax_bsf.set_title('BSF')
    if log_plot:
        ax_bsf.set_yscale('log')
    plt.plot(bsf_list, c='orange', label='BSF')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()

    #  Average changes during time
    ax_avg = plt.subplot(2, 2, 2)
    ax_avg.set_title('Average')
    if log_plot:
        ax_avg.set_yscale('log')
    plt.plot(avg_list, c='coral', label='Average')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()

    #  Average changes during time
    ax_avg = plt.subplot(2, 2, 3)
    ax_avg.set_title('Compare AVG/BSF')
    if log_plot:
        ax_avg.set_yscale('log')
    plt.plot(bsf_list, c='orange', label='BSF')
    plt.plot(avg_list, c='coral', label='Average')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()

    #  Trajectory changes during time
    ax_trajectory = plt.subplot(2, 2, 4)
    ax_trajectory.set_title('Trajectory')
    plt.plot(trajectory, c='lightgreen', label='Trajectory')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()

    #  plt.savefig(str(title) + '.png')

    # --------------------------------------------

    fig4 = plt.figure(4, figsize=(9, 7))
    fig4.suptitle('Wind/Velocity')

    # Velocity
    ax_history = plt.subplot(2, 2, 1)
    ax_history.set_title('Velocity')
    plt.plot(v_history, c='red', label='Velocity mean')

    # Wind
    ax_wind = plt.subplot(2, 2, 2)
    ax_wind.set_title('Wind')
    plt.plot(w_history, c='blue', label='Wind mean')

    # F
    ax_wind = plt.subplot(2, 2, 3)
    ax_wind.set_title('F')
    plt.plot(f_history, c='green', label='F')

    # a
    ax_wind = plt.subplot(2, 2, 4)
    ax_wind.set_title('Acceleration')
    plt.plot(a_history, c='magenta', label='a')

    plt.show(block=block)
