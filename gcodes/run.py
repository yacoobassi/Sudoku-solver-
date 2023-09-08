import subprocess

# Define the command to run the other Python script
command = ["python", "send.py", "-p", "COM11", "-f", "up.gcode", "-r", "1", "-v", "1"]

# Replace "path/to/your_script.py", "your_port", "your_file.gcode", "your_repetition", and "your_verbosity" with the actual values you want to pass as command-line arguments.

# Run the command
subprocess.run(command)
