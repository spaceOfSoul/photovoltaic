import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class PlotGenerator:
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def _decompose(self, data, period=12):
        """Decompose data into trend and seasonal components"""
        decomposition = seasonal_decompose(data, model='additive', period=period)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        return trend, seasonal

    def plot_monthly(self, y_true_chunks, result_chunks):
        for i in range(12):
            fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            y_true = np.concatenate(y_true_chunks[i])
            result = np.concatenate(result_chunks[i])

            # Decompose data
            trend_y_true, seasonal_y_true = self._decompose(y_true)
            trend_result, seasonal_result = self._decompose(result)

            # Plot original
            axes[0].plot(y_true, label='True')
            axes[0].plot(result, label='Predicted')
            axes[0].legend(loc='upper left')
            axes[0].set_title(f'Month {i+1} - Original')

            # Plot trend
            axes[1].plot(trend_y_true, label='True')
            axes[1].plot(trend_result, label='Predicted')
            axes[1].legend(loc='upper left')
            axes[1].set_title(f'Month {i+1} - Trend')

            # Plot seasonal
            axes[2].plot(seasonal_y_true, label='True')
            axes[2].plot(seasonal_result, label='Predicted')
            axes[2].legend(loc='upper left')
            axes[2].set_title(f'Month {i+1} - Seasonal')

            plt.tight_layout()
            plt.savefig(os.path.join(self.image_dir, f"month_{i+1}.png"))
            plt.close() 

    def plot_annual(self, y_true, result):
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))

        # Assuming there are 24 data points for each day
        points_per_day = 24

        # Compute average daily values
        y_true_avg = np.mean(np.array(y_true).reshape(-1, points_per_day), axis=1)
        result_avg = np.mean(np.array(result).reshape(-1, points_per_day), axis=1)

        # For the time axis
        days = np.arange(1, len(y_true_avg) + 1)

        # Decompose into trend, seasonal, and residual components
        decomposition_y_true = seasonal_decompose(y_true_avg, model='additive', period=12)
        decomposition_result = seasonal_decompose(result_avg, model='additive', period=12)

        # Plot original
        axes[0].plot(days, y_true_avg, label='True')
        axes[0].plot(days, result_avg, label='Predicted')
        axes[0].legend(loc='upper left')
        axes[0].set_title('Original')

        # Plot trend
        axes[1].plot(days, decomposition_y_true.trend, label='True')
        axes[1].plot(days, decomposition_result.trend, label='Predicted')
        axes[1].legend(loc='upper left')
        axes[1].set_title('Trend')

        # Plot seasonal
        axes[2].plot(days, decomposition_y_true.seasonal, label='True')
        axes[2].plot(days, decomposition_result.seasonal, label='Predicted')
        axes[2].legend(loc='upper left')
        axes[2].set_title('Seasonal')

        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "annual_pattern.png"))
        plt.close()  

