import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict

class TimeSeriesAnomalyInjector:
    def __init__(self, seed: int = 42):
        """
        Initialize the Anomaly Injector with a configurable random seed

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.rng = np.random.default_rng(seed)
    
    def inject_anomaly(
        self, 
        data: Union[pd.DataFrame, pd.Series],
        anomaly_settings: Optional[Dict] = None,  # Add anomaly_settings argument
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Inject anomalies into a single DataFrame or Series.
        Can inject anomalies at a specific timestamp or randomly.

        Args:
            data (DataFrame or Series): Input time series data
            anomaly_settings (Dict, optional): Settings for targeted anomaly injection.
                                                Should contain 'timestamp' and any settings
                                                needed for _inject_series_anomaly.
            columns (List[str], optional): Columns to inject anomalies into. 
                Defaults to all numeric columns for DataFrame.
            anomaly_type (str, optional): Type of anomaly to inject. 
                Choices: 'point', 'spike', 'step', 'drift'.
            percentage (float, optional): Percentage of data points to modify. 
                Defaults to 0.1.
            magnitude (float, optional): Scaling factor for anomaly intensity. 
                Defaults to 1.0.
            custom_params (Dict, optional): Additional parameters for specific anomaly types

        Returns:
            DataFrame or Series with injected anomalies
        """
        if anomaly_settings:  # If anomaly_settings are provided
            timestamp = anomaly_settings.get('timestamp')
            if timestamp is not None:
                return self._inject_anomaly_at_timestamp(data, anomaly_settings)

        # Handle both DataFrame and Series inputs
        if isinstance(data, pd.Series):
            return self._inject_series_anomaly(
                data, 
                anomaly_settings.get('anomaly_type'), 
                anomaly_settings.get('percentage'), 
                anomaly_settings.get('magnitude'), 
            )
        
        # For DataFrame
        modified_data = data.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = list(modified_data.select_dtypes(include=[np.number]).columns)
        
        # Inject anomalies for each specified column
        for column in columns:
            modified_data[column] = self._inject_series_anomaly(
                modified_data[column], 
                anomaly_settings.get('anomaly_type'), 
                anomaly_settings.get('percentage'), 
                anomaly_settings.get('magnitude'), 
            )
        
        return modified_data
    
    def _inject_series_anomaly(
        self, 
        series: pd.Series,
        anomaly_type: str,
        percentage: float,
        magnitude: float,
    ) -> pd.Series:
        """
        Inject anomalies into a single Series

        Args:
            series (pd.Series): Input time series
            anomaly_type (str): Type of anomaly to inject
            percentage (float): Percentage of data points to modify
            magnitude (float): Scaling factor for anomaly intensity
            custom_params (Dict, optional): Additional configuration

        Returns:
            pd.Series with injected anomalies
        """
        # Calculate number of anomalies to inject
        num_anomalies = max(1, int(len(series) * percentage))
        
        # Create a copy of the series to modify
        modified_series = series.copy()
        
        # Select random indices to modify
        anomaly_indices = self.rng.choice(
            series.index, 
            size=num_anomalies, 
            replace=False
        )
        
        # Anomaly injection strategies
        if anomaly_type == 'hightened':
            modifications = self.rng.normal(
                loc=0, 
                scale=series.std() * magnitude, 
                size=num_anomalies
            )
            modified_series.loc[anomaly_indices] += modifications
        
        elif anomaly_type == 'spike':
            modifications = self.rng.normal(
                loc=0, 
                scale=series.std() * (magnitude * 2), 
                size=num_anomalies
            )
            modified_series.loc[anomaly_indices] += modifications
        
        elif anomaly_type == 'lowered':

            data_range = series.max() - series.min()
            random_factors = self.rng.uniform(0.4, 0.5, size=num_anomalies) #Generate a randomized percentage between 40-50% 
            step_values = -data_range * random_factors

            modified_series.loc[anomaly_indices] = np.maximum( modified_series.loc[anomaly_indices] + step_values, 0 ) # Apply the step values but clip the result to ensure it stays >= 0
        
        elif anomaly_type == 'offline':
            modifications = self.rng.normal(
                loc=0, 
                scale=series.std() * 0.000001,
                size=num_anomalies
            )
            modified_series.loc[anomaly_indices] += modifications
        
        elif anomaly_type == 'custom':
            # Calculate the number of anomalies to inject based on percentage
            num_anomalies = max(1, int(len(series) * percentage))

            # Select random indices to modify
            anomaly_indices = self.rng.choice(
                series.index, 
                size=num_anomalies, 
                replace=False
            )

            # Apply the custom modifications to the selected indices
            modified_series.loc[anomaly_indices] *= magnitude
        
        return modified_series
    
    def _inject_anomaly_at_timestamp(self, data, anomaly_settings):
        """Injects an anomaly at a specific timestamp."""
        modified_data = data.copy()
        timestamp = anomaly_settings.pop('timestamp')  # Remove timestamp from settings

        # Assuming the first column is the timestamp column
        target_index = modified_data[modified_data.iloc[:, 0] == timestamp].index  

        if len(target_index) > 0:  # If the timestamp is found
            # Get the relevant settings from anomaly_settings
            columns = anomaly_settings.get('columns')
            anomaly_type = anomaly_settings.get('anomaly_type', 'custom')
            magnitude = anomaly_settings.get('magnitude', 1.0)

            # Inject the anomaly into the specified columns
            for column in columns:
                modified_data[column] = self._inject_series_anomaly(
                    modified_data[column],
                    anomaly_type,
                    1.0,  # Inject into 100% of the selected points (the single timestamp)
                    magnitude,
                )
                # Since we're targeting a specific timestamp, 
                # we override the percentage to 1.0 to ensure the anomaly is injected 
                # at that point. The anomaly_indices will only contain the target index.

        return modified_data