import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
import os

# --- GUI Setup ---
root = tk.Tk()
root.title("Video Helper Utilities")
root.geometry("600x480") # Increased size slightly for FPS field

# Use themed widgets for a slightly nicer look
style = ttk.Style(root)
try:
    style.theme_use('vista') # Common on Windows
except tk.TclError:
    print("Selected theme not available, using default.")


# --- Variables ---
f2v_input_dir = tk.StringVar()
f2v_output_dir = tk.StringVar()
f2v_fps = tk.StringVar(value="10.0") # Variable for FPS, with default value
mv_input_dir = tk.StringVar()
mv_output_file = tk.StringVar()

# --- Functions ---
def select_directory(target_variable, title="Select Directory"):
    """Opens a directory selection dialog and updates the target variable."""
    dir_path = filedialog.askdirectory(title=title)
    if dir_path:
        target_variable.set(dir_path)

def select_save_file(target_variable, title="Select Output Video File", defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("MP4 files", "*.mp4"), ("All files", "*.*")]):
    """Opens a file save dialog and updates the target variable."""
    file_path = filedialog.asksaveasfilename(title=title, defaultextension=defaultextension, filetypes=filetypes)
    if file_path:
        target_variable.set(file_path)

def run_script_in_thread(script_path, args_list):
    """Runs a script in a separate thread to avoid blocking the GUI."""
    def target():
        status_label.config(text=f"Running {os.path.basename(script_path)}...")
        try:
            # Ensure the script path exists
            if not os.path.exists(script_path):
                 messagebox.showerror("Error", f"Script not found: {script_path}")
                 status_label.config(text="Error: Script not found.")
                 return

            # Construct the command
            command = ["python", script_path] + args_list
            print(f"Executing command: {' '.join(command)}") # For debugging

            # Run the process
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) # Hide console window on Windows
            stdout, stderr = process.communicate()

            # Display output/errors (optional, could be logged to a file or a text widget)
            print("--- STDOUT ---")
            print(stdout)
            print("--- STDERR ---")
            print(stderr)

            if process.returncode == 0:
                status_label.config(text=f"Finished: {os.path.basename(script_path)} successfully.")
                messagebox.showinfo("Success", f"{os.path.basename(script_path)} completed successfully.")
            else:
                status_label.config(text=f"Error running {os.path.basename(script_path)}.")
                messagebox.showerror("Error", f"Error running {os.path.basename(script_path)}.\n\nError:\n{stderr}")

        except Exception as e:
            status_label.config(text="An unexpected error occurred.")
            messagebox.showerror("Execution Error", f"An error occurred: {e}")

    # Start the thread
    thread = threading.Thread(target=target, daemon=True)
    thread.start()


def run_frames_to_video():
    """Validates inputs and runs the frames_to_video script."""
    in_dir = f2v_input_dir.get()
    out_dir = f2v_output_dir.get()
    fps_str = f2v_fps.get() # Get FPS value from the GUI variable
    script_path = "frames_to_video.py" # Assumes script is in the same directory

    if not in_dir or not out_dir:
        messagebox.showwarning("Input Missing", "Please select both input and output directories for Frames-to-Video.")
        return
    if not os.path.isdir(in_dir):
         messagebox.showerror("Invalid Input", f"Input directory does not exist:\n{in_dir}")
         return

    # Validate FPS input
    try:
        fps_val = float(fps_str)
        if fps_val <= 0:
            raise ValueError("FPS must be positive")
    except ValueError:
        messagebox.showerror("Invalid Input", f"Invalid FPS value: '{fps_str}'. Please enter a positive number.")
        return

    # Pass arguments including --fps
    run_script_in_thread(script_path, [in_dir, out_dir, "--fps", str(fps_val)])


def run_merge_videos():
    """Validates inputs and runs the merge_videos script."""
    in_dir = mv_input_dir.get()
    out_file = mv_output_file.get()
    script_path = "merge_videos.py" # Assumes script is in the same directory

    if not in_dir or not out_file:
        messagebox.showwarning("Input Missing", "Please select the input directory and output file for Merge Videos.")
        return
    if not os.path.isdir(in_dir):
         messagebox.showerror("Invalid Input", f"Input directory does not exist:\n{in_dir}")
         return

    run_script_in_thread(script_path, [in_dir, out_file])


# --- GUI Layout ---

# Frames to Video Section
f2v_frame = ttk.LabelFrame(root, text="Frames to Video", padding=(10, 5))
f2v_frame.pack(padx=10, pady=(10,5), fill="x") # Adjusted padding

ttk.Label(f2v_frame, text="Input Dir (contains subfolders with images):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
f2v_input_entry = ttk.Entry(f2v_frame, textvariable=f2v_input_dir, width=50)
f2v_input_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
f2v_input_btn = ttk.Button(f2v_frame, text="Browse...", command=lambda: select_directory(f2v_input_dir, "Select Input Image Directory"))
f2v_input_btn.grid(row=0, column=2, padx=5, pady=5)

ttk.Label(f2v_frame, text="Output Dir (where videos will be saved):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
f2v_output_entry = ttk.Entry(f2v_frame, textvariable=f2v_output_dir, width=50)
f2v_output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
f2v_output_btn = ttk.Button(f2v_frame, text="Browse...", command=lambda: select_directory(f2v_output_dir, "Select Output Video Directory"))
f2v_output_btn.grid(row=1, column=2, padx=5, pady=5)

# --- FPS Input Row --- Added row index 2 ---
ttk.Label(f2v_frame, text="Frame Rate (FPS):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
f2v_fps_entry = ttk.Entry(f2v_frame, textvariable=f2v_fps, width=10) # Made FPS entry smaller
f2v_fps_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w") # Aligned left

# --- Run Button --- Moved to row index 3 ---
f2v_run_btn = ttk.Button(f2v_frame, text="Run Frames-to-Video", command=run_frames_to_video)
f2v_run_btn.grid(row=3, column=0, columnspan=3, pady=(10,5)) # Adjusted padding

f2v_frame.columnconfigure(1, weight=1) # Make entry expand

# Merge Videos Section
mv_frame = ttk.LabelFrame(root, text="Merge Videos", padding=(10, 5))
mv_frame.pack(padx=10, pady=5, fill="x") # Adjusted padding

ttk.Label(mv_frame, text="Input Dir (contains videos to merge):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
mv_input_entry = ttk.Entry(mv_frame, textvariable=mv_input_dir, width=50)
mv_input_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
mv_input_btn = ttk.Button(mv_frame, text="Browse...", command=lambda: select_directory(mv_input_dir, "Select Input Video Directory"))
mv_input_btn.grid(row=0, column=2, padx=5, pady=5)

ttk.Label(mv_frame, text="Output Video File (merged video):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
mv_output_entry = ttk.Entry(mv_frame, textvariable=mv_output_file, width=50)
mv_output_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
mv_output_btn = ttk.Button(mv_frame, text="Browse...", command=lambda: select_save_file(mv_output_file, "Select Output Merged Video File"))
mv_output_btn.grid(row=1, column=2, padx=5, pady=5)

mv_run_btn = ttk.Button(mv_frame, text="Run Merge Videos", command=run_merge_videos)
mv_run_btn.grid(row=2, column=0, columnspan=3, pady=(10,5)) # Adjusted padding

mv_frame.columnconfigure(1, weight=1) # Make entry expand

# Status Label
status_label = ttk.Label(root, text="Status: Idle", relief=tk.SUNKEN, anchor="w", padding=5)
status_label.pack(side=tk.BOTTOM, fill="x", padx=10, pady=5)

# --- Run GUI ---
root.mainloop()