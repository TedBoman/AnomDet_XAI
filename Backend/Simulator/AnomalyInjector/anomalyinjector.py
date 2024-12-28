import sys
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import datetime as dt

from Simulator.DBAPI import utils as ut
from Simulator.DBAPI.talk_to_backend import AnomalySetting
from Simulator.AnomalyInjector.InjectionMethods.lowered import LoweredAnomaly
from Simulator.AnomalyInjector.InjectionMethods.spike import SpikeAnomaly
from Simulator.AnomalyInjector.InjectionMethods.step import StepAnomaly
from Simulator.AnomalyInjector.InjectionMethods.custom import CustomAnomaly
from Simulator.AnomalyInjector.InjectionMethods.offline import OfflineAnomaly

class TimeSeriesAnomalyInjector:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def inject_anomaly(
        self, 
        data: Union[pd.DataFrame, pd.Series],
        anomaly_settings: Optional[Union[AnomalySetting, List[AnomalySetting]]] = None,
    ) -> Union[pd.DataFrame, pd.Series]:
        try:
            if isinstance(anomaly_settings, AnomalySetting):
                anomaly_settings = [anomaly_settings]

            if not anomaly_settings:
                return data

            modified_data = data.copy(deep=True)
            timestamp_col = modified_data.columns[0]

            for setting in anomaly_settings:
                try:
                    # Parse timestamps and create span mask
                    start_time = setting.timestamp
                    duration = ut.parse_duration_seconds(setting.duration)
                    columns = setting.columns
                    percentage = setting.percentage

                    modified_data[timestamp_col] = pd.to_datetime(modified_data[timestamp_col], unit='s')
                    start_time = pd.to_datetime(start_time) if isinstance(start_time, str) else start_time
                    end_time = start_time + pd.Timedelta(seconds=duration) if duration else start_time

                    span_mask = (modified_data[timestamp_col] >= start_time) & \
                                (modified_data[timestamp_col] <= end_time)
                    span_data = modified_data[span_mask]

                    # Log span details
                    print(f"Span data size: {len(span_data)}")
                    print(f"Span timestamps: {start_time} to {end_time}")
                    print(f"Percentage of anomalies to inject: {percentage}")
                    sys.stdout.flush()

                    for column in (columns or span_data.select_dtypes(include=[np.number]).columns):
                        try:
                            if column in span_data.columns:
                                data_range = (
                                    modified_data[column].max() - modified_data[column].min() 
                                    if setting.data_range is None else setting.data_range[0]
                                )
                                mean = (
                                    modified_data[column].mean() 
                                    if setting.mean is None else setting.mean[0]
                                )

                                # Log data range and mean
                                print(f"Data range: {data_range}, Mean: {mean}")
                                sys.stdout.flush()

                                num_anomalies = min(len(span_data), max(1, int(len(span_data) * percentage)))
                                anomaly_indices = self.rng.choice(span_data.index, size=num_anomalies, replace=False)
                                print(f"Anomaly indices selected: {anomaly_indices}")
                                print(f"Data before injection:\n{modified_data.loc[anomaly_indices, column]}")
                                sys.stdout.flush()

                                try:
                                    # Apply anomaly injection
                                    modified_data.loc[anomaly_indices, column] = self._apply_anomaly(
                                        modified_data.loc[anomaly_indices, column],
                                        data_range,
                                        self.rng,
                                        mean,
                                        setting
                                    )
                                    # Mark anomaly flags
                                    modified_data.loc[anomaly_indices, 'injected_anomaly'] = True
                                    print(f"Data after injection:\n{modified_data.loc[anomaly_indices, column]}")
                                    print(f"Anomaly flags after injection:\n{modified_data.loc[anomaly_indices, 'injected_anomaly']}")
                                    sys.stdout.flush()
                                except Exception as e:
                                    print(f"Error during anomaly application for column {column}: {e}")
                                    sys.stdout.flush()
                        except Exception as e:
                            print(f"Error processing column {column}: {e}")
                            sys.stdout.flush()
                except Exception as e:
                    print(f"Error processing anomaly setting: {e}")
                    sys.stdout.flush()

            return modified_data
        except Exception as e:
            print(f"Error in inject_anomaly: {e}")
            sys.stdout.flush()
            return data

    def _apply_anomaly(self, data, data_range, rng, mean, settings: AnomalySetting):
        try:
            anomaly_type = settings.anomaly_type
            magnitude = settings.magnitude

            # Log data before applying anomaly
            print(f"Old data before anomaly application: {data}")
            sys.stdout.flush()

            if anomaly_type == 'lowered':
                injector = LoweredAnomaly()
                result = injector.inject_anomaly(data, rng, data_range, mean)
                print(f"New data after lowered anomaly: {result}")
                sys.stdout.flush()
                return result

            elif anomaly_type == 'spike':
                injector = SpikeAnomaly()
                result = injector.inject_anomaly(data, rng, magnitude)
                print(f"New data after spike anomaly: {result}")
                sys.stdout.flush()
                return result

            elif anomaly_type == 'step':
                injector = StepAnomaly()
                result = injector.inject_anomaly(data, mean, magnitude)
                print(f"New data after step anomaly: {result}")
                sys.stdout.flush()
                return result

            elif anomaly_type == 'offline':
                injector = OfflineAnomaly()
                result = injector.inject_anomaly()
                print(f"New data after offline anomaly: {result}")
                sys.stdout.flush()
                return result

            elif anomaly_type == 'custom':
                injector = CustomAnomaly()
                result = injector.inject_anomaly(data, magnitude)
                print(f"New data after custom anomaly: {result}")
                sys.stdout.flush()
                return result

            return data
        except Exception as e:
            print(f"Error in _apply_anomaly: {e}")
            sys.stdout.flush()
            return data
