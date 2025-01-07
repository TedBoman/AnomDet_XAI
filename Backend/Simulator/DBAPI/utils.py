from datetime import timedelta
import datetime
import re
import sys

def parse_duration(duration_str):
        """
        Parses a duration string like '1H', '30min', '2D', '1h30m', '2days 5hours' 
        into a timedelta object.
        
        Supports the following units:
            - H, h: hours
            - min, m: minutes
            - D, d, days: days
            - S, s: seconds
            - W, w, weeks: weeks

        Args:
            duration_str (str): The duration string to parse.

        Returns:
            datetime.timedelta: A timedelta object representing the duration.

        Raises:
            ValueError: If the duration string is invalid.
        """

        if duration_str == "0" or duration_str == None or duration_str == "":
            return timedelta(0)

        pattern = r'(\d+)\s*([HhmindaysSwW]+)'
        matches = re.findall(pattern, duration_str)

        if not matches:
            raise ValueError("Invalid duration format")

        total_seconds = 0
        for value, unit in matches:
            value = int(value)
            if unit in ('H', 'h'):
                total_seconds += value * 3600
            elif unit in ('min', 'm'):
                total_seconds += value * 60
            elif unit in ('D', 'd', 'days'):
                total_seconds += value * 86400
            elif unit in ('S', 's'):
                total_seconds += value
            elif unit in ('W', 'w', 'weeks'):
                total_seconds += value * 604800
            else:
                return timedelta(0)

        return timedelta(seconds=total_seconds)

def parse_duration_seconds(duration_str):
        """
        Parses a duration string like '1H', '30min', '2D', '1h30m', '2days 5hours' 
        into a timedelta object.
        
        Supports the following units:
            - H, h: hours
            - min, m: minutes
            - D, d, days: days
            - S, s: seconds
            - W, w, weeks: weeks

        Args:
            duration_str (str): The duration string to parse.

        Returns:
            datetime.timedelta: A timedelta object representing the duration.

        Raises:
            ValueError: If the duration string is invalid.
        """

        if duration_str == "0" or duration_str == None or duration_str == "":
            return 0

        pattern = r'(\d+)\s*([HhmindaysSwW]+)'
        matches = re.findall(pattern, duration_str)

        if not matches:
            raise ValueError("Invalid duration format")

        total_seconds = 0
        for value, unit in matches:
            value = int(value)
            if unit in ('H', 'h'):
                total_seconds += value * 3600
            elif unit in ('min', 'm'):
                total_seconds += value * 60
            elif unit in ('D', 'd', 'days'):
                total_seconds += value * 86400
            elif unit in ('S', 's'):
                total_seconds += value
            elif unit in ('W', 'w', 'weeks'):
                total_seconds += value * 604800
            else:
                return 0

        return total_seconds - 30