from matplotlib import pyplot as plt
import numpy as np 

def plot_loss_curve(losses_A, losses_B=None, title=""):

    plt.figure(figsize=(10,3))

    if isinstance(losses_B, type(None)):
        plt.plot(losses_A, color="black")
    else:
        plt.plot(losses_A, color="blue", linewidth=1, label="No Forward Masks")
        plt.plot(losses_B, color="red", linewidth=1, label="Forward Masks", alpha=0.5)
        plt.legend()

    plt.yscale('log')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    
def plot_loss_logZ_curves(losses, logZs):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    plt.sca(ax[0])
    plt.plot(losses, color="black")
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.sca(ax[1])
    plt.plot(np.exp(logZs), color="black")
    plt.ylabel('Estimated Z')
    plt.xlabel('Step')
    plt.suptitle("Loss and Estimated Partition Function for the Trajectory Balance Model")
