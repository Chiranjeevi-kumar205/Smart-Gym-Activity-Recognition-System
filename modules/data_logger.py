import pandas as pd
from datetime import datetime
import os

class DataLogger:
    """
    Logs activity data to a CSV file.
    """
    def __init__(self, log_file):
        """
        Initializes the logger. Creates the log file with a header if it doesn't exist.
        """
        self.log_file = log_file
        self.columns = ['Timestamp', 'PersonID', 'Activity']
        
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

    def log(self, person_id, activity):
        """
        Appends a new activity record to the log file.

        Args:
            person_id (int): The ID of the person.
            activity (str): The classified activity.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_log = pd.DataFrame([[timestamp, person_id, activity]], columns=self.columns)
        new_log.to_csv(self.log_file, mode='a', header=False, index=False)
        print(f"Logged: Person {person_id} - {activity}")