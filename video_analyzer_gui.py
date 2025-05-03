import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font
import threading
import subprocess
import os

# --- Constants & Style ---
BG_COLOR = "#2E3F4F" # A slightly muted dark blue/grey
FG_COLOR = "#EAEAEA" # Off-white text
BTN_BG_COLOR = "#5D6D7E"
BTN_FG_COLOR = "#FFFFFF"
BTN_ACTIVE_BG_COLOR = "#85929E"
LABEL_FG_COLOR = "#BDC3C7" # Slightly dimmer text for non-critical labels
STATUS_FG_SUCCESS = "#2ECC71" # Green
STATUS_FG_ERROR = "#E74C3C" # Red
STATUS_FG_WARN = "#F1C40F" # Yellow
STATUS_FG_INFO = "#FFFFFF" # White for status updates

# --- GUI window setup ---
root = tk.Tk()
root.title("Video Analyzer")
root.geometry("550x400")  # Slightly wider
root.configure(bg=BG_COLOR)

# --- Style Configuration ---
style = ttk.Style(root)
try:
    # Try themes for a modern look, 'clam' is often available cross-platform
    selected_theme = 'clam' if 'clam' in style.theme_names() else 'default'
    style.theme_use(selected_theme)
    print(f"Using theme: {selected_theme}")
except tk.TclError:
    print("Selected theme not available, using default.")

# Custom font
default_font = font.nametofont("TkDefaultFont")
default_font.configure(size=10)
bold_font = font.Font(family=default_font['family'], size=12, weight='bold')
status_font = font.Font(family=default_font['family'], size=9, slant='italic')

# Configure ttk styles
style.configure("TFrame", background=BG_COLOR)
style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=default_font)
style.configure("Header.TLabel", foreground=FG_COLOR, font=bold_font, anchor=tk.CENTER)
style.configure("Selected.TLabel", foreground=LABEL_FG_COLOR, font=default_font, wraplength=500, anchor=tk.W) # Wraps long paths
style.configure("Status.TLabel", font=status_font, padding=5, anchor=tk.W)
style.configure("TLabelframe", background=BG_COLOR, relief=tk.GROOVE, borderwidth=1)
style.configure("TLabelframe.Label", background=BG_COLOR, foreground=FG_COLOR, font=default_font)

style.configure("TButton",
                background=BTN_BG_COLOR,
                foreground=BTN_FG_COLOR,
                font=default_font,
                padding=(10, 5), # Hoz, Vert padding
                borderwidth=1,
                relief=tk.RAISED)
style.map("TButton",
          background=[('active', BTN_ACTIVE_BG_COLOR), ('pressed', BTN_ACTIVE_BG_COLOR)],
          relief=[('pressed', tk.SUNKEN)])


# --- Global Variables ---
selected_video = tk.StringVar()

# --- Functions ---

def select_video():
    # Use initialdir for convenience if desired
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    if file_path:
        selected_video.set(file_path)
        # Display only the filename for brevity, full path in variable
        filename = os.path.basename(file_path)
        label_selected.config(text=f"Selected: {filename}", style="Selected.TLabel")
    else:
        selected_video.set("")
        label_selected.config(text="No file selected", style="Selected.TLabel")


def run_script(script_name):
    video_file = selected_video.get()
    if not video_file:
        label_status.config(text="Please select a video first!", foreground=STATUS_FG_ERROR, style="Status.TLabel")
        messagebox.showwarning("Input Missing", "Please select a video file before running an analysis.")
        return

    if not os.path.exists(script_name):
        error_msg = f"Error: Script '{script_name}' not found!"
        label_status.config(text=error_msg, foreground=STATUS_FG_ERROR, style="Status.TLabel")
        messagebox.showerror("Script Not Found", f"The script file '{script_name}' was not found in the current directory.")
        return

    status_msg = f"Processing '{os.path.basename(video_file)}' with {script_name}..."
    label_status.config(text=status_msg, foreground=STATUS_FG_WARN, style="Status.TLabel")
    root.update_idletasks() # Ensure status updates immediately

    command = ["python", script_name, video_file]

    def target():
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                print(f"--- {script_name} STDOUT ---\n{stdout}")
                # Safely update GUI from the main thread
                root.after(0, lambda: label_status.config(text=f"Finished: {os.path.basename(script_name)} successfully.", foreground=STATUS_FG_SUCCESS, style="Status.TLabel"))
                # Optionally show success message box
                # root.after(0, lambda: messagebox.showinfo("Success", f"{os.path.basename(script_name)} completed."))
            else:
                print(f"--- {script_name} STDERR ---\n{stderr}")
                root.after(0, lambda: label_status.config(text=f"Error running {os.path.basename(script_name)}. Check logs.", foreground=STATUS_FG_ERROR, style="Status.TLabel"))
                # Show truncated error in message box
                error_summary = stderr[:500] + ('...' if len(stderr) > 500 else '')
                root.after(0, lambda: messagebox.showerror("Script Error", f"Error running {os.path.basename(script_name)}.\n\nError:\n{error_summary}"))

        except Exception as e:
             print(f"Error launching subprocess for {script_name}: {e}")
             root.after(0, lambda: label_status.config(text=f"Failed to launch {script_name}.", foreground=STATUS_FG_ERROR, style="Status.TLabel"))
             root.after(0, lambda: messagebox.showerror("Launch Error", f"Could not launch the script {script_name}.\nError: {e}"))

    # Run the target function in a separate thread
    thread = threading.Thread(target=target, daemon=True)
    thread.start()

# --- UI Layout ---

# Main Title
label_title = ttk.Label(root, text="Video Analyzer", style="Header.TLabel")
label_title.pack(pady=(15, 10), fill="x") # More padding top

# Frame for File Selection
frame_select = ttk.Frame(root, padding=(10, 5))
frame_select.pack(fill="x", padx=20)

btn_select = ttk.Button(frame_select, text="Choose Video File", command=select_video)
btn_select.pack(side=tk.LEFT, padx=(0, 10))

label_selected = ttk.Label(frame_select, text="No file selected", style="Selected.TLabel")
label_selected.pack(side=tk.LEFT, fill="x", expand=True)

# Frame for Analysis Buttons (using LabelFrame for grouping)
frame_buttons = ttk.LabelFrame(root, text="Analysis Options", padding=(15, 10))
frame_buttons.pack(pady=20, padx=20, fill="x")

# Configure grid columns to expand equally
frame_buttons.columnconfigure(0, weight=1)
frame_buttons.columnconfigure(1, weight=1)
frame_buttons.columnconfigure(2, weight=1)

btn_presence = ttk.Button(
    frame_buttons, text="Presence Heatmap", command=lambda: run_script("presence_heatmap.py")
)
# Use sticky='ew' to make buttons fill their grid cell horizontally
btn_presence.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

btn_speed_tracker = ttk.Button(
    frame_buttons, text="Speed Tracker", command=lambda: run_script("speed_tracker.py")
)
btn_speed_tracker.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

btn_speed_heatmap = ttk.Button(
    frame_buttons, text="Speed Heatmap", command=lambda: run_script("speed_heatmap.py")
)
btn_speed_heatmap.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

# Status Label (like a status bar)
label_status = ttk.Label(root, text="Status: Idle", style="Status.TLabel", foreground=STATUS_FG_INFO, relief=tk.SUNKEN)
label_status.pack(side=tk.BOTTOM, fill="x", pady=(10, 0), ipady=2) # ipady adds internal padding

# --- Run GUI ---
root.mainloop()