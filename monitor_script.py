import os
import time
import subprocess
from datetime import datetime

LOG_FILE = "./output.log"  # Path to your log file
SCRIPT_PATH = "./en_fr_summaries-v2.py"  # Path to your Python script
TIMEOUT = 60  # Timeout in seconds (e.g., 5 minutes)

def get_last_log_timestamp():
    """Extract the timestamp of the last log entry."""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, 'r') as log_file:
        lines = log_file.readlines()
        if not lines:
            return None
        last_line = lines[-1]
        try:
            timestamp = " ".join(last_line.split()[:2])
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
        except Exception as e:
            print(f"Error parsing timestamp: {e}")
            return None

def is_script_stalled(last_timestamp):
    """Check if the script is stalled."""
    if not last_timestamp:
        return True
    now = datetime.now()
    delta = (now - last_timestamp).total_seconds()
    return delta > TIMEOUT

def restart_script():
    """Restart the main Python script."""
    print("Script stalled. Restarting...")
    subprocess.run(["pkill", "-f", SCRIPT_PATH])  # Kill the script
    time.sleep(5)  # Wait before restarting
    subprocess.Popen(["python3", SCRIPT_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def monitor_script():
    subprocess.Popen(["python3", SCRIPT_PATH], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    """Monitor and restart the script if needed."""
    while True:
        last_timestamp = get_last_log_timestamp()
        if is_script_stalled(last_timestamp):
            restart_script()
        time.sleep(60)  # Check every 60 seconds

if __name__ == "__main__":
    monitor_script()
