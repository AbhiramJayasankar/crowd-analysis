import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import threading
import subprocess
from tkinter import ttk

# GUI window setup
root = tk.Tk()
root.title("Heatmap Video Processor")
root.geometry("500x400")
root.configure(bg="#2C3E50")

selected_video = tk.StringVar()


# Function to choose video file
def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if file_path:
        selected_video.set(file_path)
        label_selected.config(text=f"Selected: {file_path}", fg="#ECF0F1")


# Function to run selected script
def run_script(script_name):
    if not selected_video.get():
        label_status.config(text="Please select a video first!", fg="#E74C3C")
        return

    label_status.config(text=f"Processing {script_name}...", fg="#F1C40F")
    threading.Thread(
        target=subprocess.run,
        args=(["python", script_name, selected_video.get()],),
        daemon=True,
    ).start()


# UI elements
label_title = tk.Label(
    root,
    text="Select a Video File and Apply Heatmap",
    font=("Arial", 14, "bold"),
    bg="#2C3E50",
    fg="#ECF0F1",
)
label_title.pack(pady=15)

btn_select = ttk.Button(root, text="Choose Video", command=select_video)
btn_select.pack(pady=10)

label_selected = tk.Label(root, text="No file selected", fg="#BDC3C7", bg="#2C3E50")
label_selected.pack()

frame_buttons = tk.Frame(root, bg="#2C3E50")
frame_buttons.pack(pady=10)

btn_one = ttk.Button(
    frame_buttons, text="Run One.py", command=lambda: run_script("one.py")
)
btn_one.grid(row=0, column=0, padx=10, pady=5)

btn_speed = ttk.Button(
    frame_buttons, text="Run Speed.py", command=lambda: run_script("speed.py")
)
btn_speed.grid(row=0, column=1, padx=10, pady=5)

btn_speed2 = ttk.Button(
    frame_buttons, text="Run Speed2.py", command=lambda: run_script("speed2.py")
)
btn_speed2.grid(row=0, column=2, padx=10, pady=5)

label_status = tk.Label(
    root, text="", fg="#ECF0F1", bg="#2C3E50", font=("Arial", 10, "italic")
)
label_status.pack(pady=20)

root.mainloop()
