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
        anomaly_settings: Optional[Dict] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Inject anomalies into a time series at a specific point or within a span.

        Args:
            data (DataFrame or Series): Input time series data
            anomaly_settings (Dict): Settings for targeted anomaly injection
                - timestamp: Specific time or start of the anomaly span
                - duration (optional): Length of the anomaly span
                - columns: Columns to inject anomalies into
                - anomaly_type: Type of anomaly ('lowered', 'spike', etc.)
                - percentage: Percentage of data points to modify (for span)
                - magnitude: Scaling factor for anomaly intensity

        Returns:
            DataFrame or Series with injected anomalies
        """
        # Validate required settings
        if not anomaly_settings:
            return data

        # Extract settings with defaults
        start_time = anomaly_settings.get('timestamp')
        duration = anomaly_settings.get('duration', None)
        columns = anomaly_settings.get('columns', [])
        anomaly_type = anomaly_settings.get('anomaly_type', 'custom')
        percentage = anomaly_settings.get('percentage', 0.1)
        magnitude = anomaly_settings.get('magnitude', 1.0)

        # Prepare modified data
        modified_data = data.copy()
        timestamp_col = modified_data.columns[0]  # Assume first column is timestamp

        # Convert timestamp column to pandas Timestamp and then to float (seconds since Unix epoch)
        modified_data[timestamp_col] = pd.to_datetime(modified_data[timestamp_col])  # Ensure it's a Timestamp
        modified_data[timestamp_col] = modified_data[timestamp_col].astype(np.int64) / 1e9  # Convert to float (seconds)

        # Ensure start_time is a pandas Timestamp and convert it to float (seconds since Unix epoch)
        start_time = pd.to_datetime(start_time) if isinstance(start_time, str) else start_time
        start_time = start_time.timestamp()  # Convert to float (seconds since Unix epoch)

        # Determine injection method based on duration
        if duration:
            # Span-based anomaly injection
            span_mask = (modified_data[timestamp_col] >= start_time) & \
                        (modified_data[timestamp_col] < start_time + pd.Timedelta(duration).total_seconds())
            span_data = modified_data[span_mask]
            
            # If no columns specified, use all numeric columns
            if not columns:
                columns = list(span_data.select_dtypes(include=[np.number]).columns)
            
            # Inject anomalies for each specified column in the span
            for column in columns:
                data_range = modified_data[column].max() - modified_data[column].min()
                mean = modified_data[column].mean()
                print(f"MAX: {column} {modified_data[column].max()}. MIN: {column} {modified_data[column].min()}")
                if column in span_data.columns:
                    print(f"Injecting anomalies into column: {column}")
                    # Create a mask for the specific column in the span
                    col_data = span_data[column]
                    
                    # Calculate number of anomalies to inject
                    num_anomalies = min(len(col_data), max(1, int(len(col_data) * percentage)))
                    
                    # Select random indices to modify within the span
                    if num_anomalies > 0:
                        print(f"Injecting {num_anomalies} anomalies.")
                        anomaly_indices = self.rng.choice(
                            col_data.index, 
                            size=num_anomalies, 
                            replace=False
                        )

                        print(f"Selected indices for anomalies: {anomaly_indices}")
                        
                        # Apply anomaly based on type
                        modified_data.loc[anomaly_indices, column] = self._apply_anomaly(
                            modified_data.loc[anomaly_indices, column], 
                            data_range,
                            mean,
                            anomaly_type, 
                            magnitude
                        )

                        print(f"Modified data for column {column}:")
                        print(modified_data.loc[anomaly_indices, column])  # Verify changes
        else:
            # Point-specific anomaly injection
            # Identify the exact timestamp
            point_mask = modified_data[timestamp_col] == start_time
            point_data = modified_data[point_mask]
            
            # If no columns specified, use all numeric columns
            if not columns:
                columns = list(point_data.select_dtypes(include=[np.number]).columns)
            
            # Inject anomalies for each specified column at the specific point
            for column in columns:
                if column in point_data.columns:
                    # Modify the specific point
                    modified_data.loc[point_mask, column] = self._apply_anomaly(
                        modified_data.loc[point_mask, column], 
                        data_range,
                        mean,
                        anomaly_type, 
                        magnitude
                    )
        
        return modified_data

    def _apply_anomaly(self, data, data_range, mean, anomaly_type, magnitude):
        """
        Apply a specific type of anomaly to the data.

        Args:
            data (pd.Series): Data to modify
            anomaly_type (str): Type of anomaly to apply
            magnitude (float): Intensity of the anomaly

        Returns:
            pd.Series: Modified data
        """
        if anomaly_type == 'lowered':
            print("Injecting lowerd anomaly!")
            random_factors = self.rng.uniform(0.3, 0.4)
            step_value = -data_range * random_factors
            print(f"Step: {step_value} = -datarange: -{data_range} * random: {random_factors} * magnitude: {magnitude}")
            
            print(f"return: {np.maximum(data + step_value, 0)}")

            return np.maximum(data + step_value, 0)
        
        elif anomaly_type == 'spike':
            print("Injecting spike anomaly!")
            std_dev = data.std()
            modifications = self.rng.normal(
                loc=0, 
                scale=std_dev * (magnitude * 2)
            )
            return data + modifications
        
        elif anomaly_type == 'step':
            print("Injecting step anomaly!")
            step_value = mean * magnitude
            return data + step_value
        
        elif anomaly_type == 'custom':
            print("Injecting custom anomaly!")
            return data * magnitude
        
        else:
            return data