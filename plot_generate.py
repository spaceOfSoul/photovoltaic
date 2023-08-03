import os
import numpy as np
from scipy import integrate
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# [training set] power per 365 days: 98318.375 [kW per 365 days]  
# 98318.375 / 365 = 269.36 kW per day  

# [test set] power per 243 days: 51449.992 [kW per 243 days] 
# 51449.992 / 243 = 211.72 kW per day  
    
# Number of days in y_true_avg: 243
# Number of days in result_avg: 243

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
            axes[0].plot(y_true, label='Ground Truth')
            axes[0].plot(result, label='Predicted')
            axes[0].legend(loc='upper left')
            axes[0].set_title(f'Month {i+1} - Original')
            axes[0].set_ylabel('[kWh]')
        
            # Plot trend
            axes[1].plot(trend_y_true, label='Ground Truth')
            axes[1].plot(trend_result, label='Predicted')
            axes[1].legend(loc='upper left')
            axes[1].set_title(f'Month {i+1} - Trend')
            axes[1].set_ylabel('[kWh]')
            
            # Plot seasonal
            axes[2].plot(seasonal_y_true, label='Ground Truth')
            axes[2].plot(seasonal_result, label='Predicted')
            axes[2].legend(loc='upper left')
            axes[2].set_title(f'Month {i+1} - Seasonal')
            axes[2].set_ylabel('[kWh]')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.image_dir, f"month_{i+1}.png"))
            plt.close() 

    def plot_annual(self, y_true, result):
        fig, axes = plt.subplots(4, 1, figsize=(30, 40))  # Subplot count increased to 4
        # fig.suptitle("2022.Jan ~ Aug", fontsize=16)

        # Assuming there are 24 data points for each day
        points_per_day = 24

        # Compute daily total values
        # y_true_sum and result_sum represent the total kW produced per day
        y_true_sum = np.sum(np.array(y_true).reshape(-1, points_per_day), axis=1)
        result_sum = np.sum(np.array(result).reshape(-1, points_per_day), axis=1)

        # Print number of elements in y_true_sum and result_sum
        logging.info(f"Number of days in y_true_sum: {len(y_true_sum)}")
        logging.info(f"Number of days in result_sum: {len(result_sum)}")

        # Compute integrals
        integral_y_true_sum = integrate.simps(y_true_sum)
        integral_result_sum = integrate.simps(result_sum)

        # For the time axis
        days = np.arange(1, len(y_true_sum) + 1)

        # Decompose into trend, seasonal, and residual components
        decomp_y_true = seasonal_decompose(y_true_sum, model='additive', period=12)
        decomp_result = seasonal_decompose(result_sum, model='additive', period=12)

        # norm factor: 전체 면적을 10000으로 정규화
        norm_factor_true = 10000/integral_y_true_sum
        norm_factor_pred = 10000/integral_result_sum

        # Plot original
        axes[0].plot(days, y_true_sum, label='Ground Truth')
        axes[0].plot(days, result_sum, label='Predicted')
        axes[0].legend(loc='upper left')
        # Add text to the title
        title_text = (f'Daily PV Power Output\n'
              f'Ground Truth Sum: {integral_y_true_sum:.2f} [kW]\n'
              f'Predicted: {integral_result_sum:.2f} [kW]')

        axes[0].set_title(title_text)
        axes[0].set_ylabel('kW per day')
        axes[0].set_xlabel('Days')

        # Plot trend
        axes[1].plot(days, decomp_y_true.trend, label='Ground Truth')
        axes[1].plot(days, decomp_result.trend, label='Predicted')
        axes[1].legend(loc='upper left')
        axes[1].set_title('Trend (2022.Jan ~ Aug)')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('kW per day')
    
        # Plot trend normalized
        axes[2].plot(days, norm_factor_true*(decomp_y_true.trend), label='Normalized Ground Truth')
        axes[2].plot(days, norm_factor_pred*(decomp_result.trend), label='Normalized Predicted')
        axes[2].legend(loc='upper left')
        axes[2].set_title('Normalized Trend [10000 kW per (2022.Jan ~ Aug)]')
        axes[2].set_xlabel('Days')
        axes[2].set_ylabel('kW per day')

        # Plot seasonal
        axes[3].plot(days, decomp_y_true.seasonal, label='Ground Truth')
        axes[3].plot(days, decomp_result.seasonal, label='Predicted')
        axes[3].legend(loc='upper left')
        axes[3].set_title('Seasonal (2022.Jan ~ Aug)')
        axes[3].set_xlabel('Days')
        axes[3].set_ylabel('kW per day')
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.image_dir, "annual_pattern.png"))
        plt.close()


