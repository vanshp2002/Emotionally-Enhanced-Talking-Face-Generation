import tkinter as tk
from tkinter import filedialog
import subprocess

# Global variable to store the process
running_process = None

# Function to execute the Python script based on category
def run_script():
    global running_process
    image_path = image_path_entry.get()
    speech_path = speech_path_entry.get()

    command = f"python generate.py -im {image_path} -is {speech_path} -m ./model/ -o ./results/"
    if running_process:
        running_process.terminate()

    # Execute the command
    running_process = subprocess.Popen(command, shell=True)

# Function to stop the running script
def stop_script():
    global running_process
    if running_process:
        running_process.terminate()

# Create the main window
window = tk.Tk()
window.title("Script Runner")

# Create and place widgets on the window
image_label = tk.Label(window, text="Image File:")
image_label.pack()
image_path_entry = tk.Entry(window)
image_path_entry.pack()
image_browse_button = tk.Button(window, text="Browse", command=lambda: image_path_entry.insert(0, filedialog.askopenfilename()))
image_browse_button.pack()

speech_label = tk.Label(window, text="Speech File:")
speech_label.pack()
speech_path_entry = tk.Entry(window)
speech_path_entry.pack()
speech_browse_button = tk.Button(window, text="Browse", command=lambda: speech_path_entry.insert(0, filedialog.askopenfilename()))
speech_browse_button.pack()

run_button = tk.Button(window, text="Run Script", command=run_script)
run_button.pack()

stop_button = tk.Button(window, text="Stop Script", command=stop_script)
stop_button.pack()

# Start the GUI main loop
window.mainloop()
