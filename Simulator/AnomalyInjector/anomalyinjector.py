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
        columns: Optional[List[str]] = None,
        anomaly_type: str = 'point',
        percentage: float = 0.1,
        magnitude: float = 1.0,
        custom_params: Optional[Dict] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Inject anomalies into a single DataFrame or Series

        Args:
            data (DataFrame or Series): Input time series data
            columns (List[str], optional): Columns to inject anomalies into. 
                Defaults to all numeric columns for DataFrame.
            anomaly_type (str, optional): Type of anomaly to inject. 
                Choices: 'point', 'spike', 'step', 'drift'
            percentage (float, optional): Percentage of data points to modify. 
                Defaults to 0.1.
            magnitude (float, optional): Scaling factor for anomaly intensity. 
                Defaults to 1.0.
            custom_params (Dict, optional): Additional parameters for specific anomaly types

        Returns:
            DataFrame or Series with injected anomalies
        """
        # Handle both DataFrame and Series inputs
        if isinstance(data, pd.Series):
            return self._inject_series_anomaly(
                data, 
                anomaly_type, 
                percentage, 
                magnitude, 
                custom_params
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
                anomaly_type, 
                percentage, 
                magnitude, 
                custom_params
            )
        
        return modified_data
    
    def _inject_series_anomaly(
        self, 
        series: pd.Series,
        anomaly_type: str,
        percentage: float,
        magnitude: float,
        custom_params: Optional[Dict] = None
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
        if anomaly_type == 'point':
            # Random spike around the current value
            modifications = self.rng.normal(
                loc=0, 
                scale=series.std() * magnitude, 
                size=num_anomalies
            )
            modified_series.loc[anomaly_indices] += modifications
        
        elif anomaly_type == 'spike':
            # More extreme point anomalies
            modifications = self.rng.normal(
                loc=0, 
                scale=series.std() * (magnitude * 2), 
                size=num_anomalies
            )
            modified_series.loc[anomaly_indices] += modifications
        
        elif anomaly_type == 'step':
            # Sudden step change
            step_values = series.std() * magnitude * self.rng.choice([-1, 1], size=num_anomalies)
            modified_series.loc[anomaly_indices] += step_values
        
        elif anomaly_type == 'drift':
            # Gradual drift from original value
            drift_direction = self.rng.choice([-1, 1], size=num_anomalies)
            drift_amount = series.std() * magnitude * np.linspace(0, 1, num_anomalies)
            
            for idx, (index, direction, amount) in enumerate(zip(
                anomaly_indices, 
                drift_direction, 
                drift_amount
            )):
                # Create a gradual drift from the original point
                modified_series.loc[index:] += direction * amount
        
        elif anomaly_type == 'custom' and custom_params:
            # Allow for completely custom anomaly injection
            custom_func = custom_params.get('func')
            if custom_func and callable(custom_func):
                modified_series = custom_func(
                    modified_series, 
                    anomaly_indices, 
                    magnitude, 
                    **custom_params.get('kwargs', {})
                )
        
        return modified_series

def anomaly_listener(input_queue, output_queue, injector=None):
    """
    Listener function to process anomaly injection requests

    Args:
        input_queue (mp.Queue): Queue to receive anomaly injection requests
        output_queue (mp.Queue): Queue to send modified data
        injector (TimeSeriesAnomalyInjector, optional): Predefined injector instance
    """
    if injector is None:
        injector = TimeSeriesAnomalyInjector()
    
    while True:
        try:
            # Receive data and anomaly configuration
            data, config = input_queue.get()
            
            if data is None:  # Sentinel to stop the process
                break
            
            # Inject anomaly
            modified_data = injector.inject_anomaly(
                data, 
                **config
            )
            
            # Put modified data in output queue
            output_queue.put(modified_data)
        
        except Exception as e:
            # Put any errors back in the queue
            output_queue.put(e)