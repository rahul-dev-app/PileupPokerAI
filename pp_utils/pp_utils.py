import numpy as np
import matplotlib.pyplot as plt

class PP_Utils:

    def __init__(self):
        pass

    def plot_probs_heatmap(self, probs):
        '''
        Plots the probabilities of every card in hand as a heat map on the grid
        '''
        if len(probs) != 80:
            raise ValueError("The list of probabilities must contain exactly 80 elements.")

        # Convert the list into a 5x4x4 NumPy array
        data = np.array(probs).reshape((5, 4, 4))

        # Determine the min and max probabilities for consistent color scaling
        vmin = np.min(data)
        vmax = np.max(data)

        # vmin = 0.0
        # vmax = 1.0

        # Create a figure and subplots
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # 1 row, 5 columns

        for idx in range(5):
            ax = axs[idx]
            im = ax.imshow(data[idx], cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(f'Action Probabilities for card {idx+1}')
            # Optionally, add probability values on the heatmap
            for (i, j), val in np.ndenumerate(data[idx]):
                ax.text(j, i, f'{val:.4f}', ha='center', va='center', color='white')
            # Remove axis labels for clarity
            ax.set_xticks([])
            ax.set_yticks([])

        # Add a single colorbar to the figure
        cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        cbar.set_label('Probability')

        # Adjust layout
        plt.tight_layout()

        # Display the heatmaps
        plt.show()