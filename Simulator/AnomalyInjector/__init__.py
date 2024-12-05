import pandas as pd
import numpy as np
from typing import Union, List, Dict, Callable

class TimeSeriesAnomalyInjector:
    def __init__(self, seed: int = 42):
        """
        Initialize the Anomaly Injector with default random seed for reproducibility
        """
        np.random.seed(seed)

    def inject_anomalies(
            self, 
        data: Union[pd.DataFrame, pd.core.groupby.DataFrameGroupBy],
        columns: Optional[List[str]] = None,
        anomaly_type: str = 'point',
        percentage: float = 0.1,
        magnitude: float = 1.0,
        chunk_mode: bool = False
    ) -> Union[pd.DataFrame, pd.core.groupby.DataFrameGroupBy]:
            """
        Inject anomalies into time series data - supports both DataFrame and chunks

        Args:
            data (DataFrame or DataFrameGroupBy): Input time series data
            columns (List[str], optional): Columns to inject anomalies into. 
                Defaults to numeric columns.
            anomaly_type (str, optional): Type of anomaly. 
                Choices: 'point', 'contextual', 'collective'. Defaults to 'point'.
            percentage (float, optional): Percentage of data points to modify. 
                Defaults to 0.1.
            magnitude (float, optional): Scaling factor for anomaly intensity. 
                Defaults to 1.0.
            chunk_mode (bool, optional): Whether input is a chunk/group. 
                Defaults to False.

        Returns:
            DataFrame or DataFrameGroupBy with injected anomalies
        """
            # If no columns specified, default to numeric columns
            if columns is None:
                columns = list(data.select_dtypes(include=[np.number]).columns)

            # Handle different input types
            if chunk_mode:
                return self._inject_chunk_anomalies(
                    data, columns, anomaly_type, percentage, magnitude
                )
            else:
                return self._inject_dataframe_anomalies(
                    data, columns, anomaly_type, percentage, magnitude
                )