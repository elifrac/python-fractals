"""
Main entry point for Fractal Explorer application.
"""
import tkinter as tk
from tkinter import ttk
import os
import shutil
import subprocess
from fractals_ui import (
    set_ui_globals,
    start_drawing_thread,
    end_drawing,
    start_zoom_rect,
    update_zoom_rect,
    finish_zoom_rect,
    load_default_parameters,
    show_context_menu,
    get_fractal_type,
    get_halley_power
)
from fractals_io import (
    save_fractal_image,
    load_fractal_image,
    save_parameters_file,
    load_parameters_file,
    load_palette
)


# Global variable to track the currently loaded parameter file
loaded_parameter_file = None


def update_parameter_display(param_display_labels, width_entry_var, height_entry_var, 
                            max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                            y_min_entry_var, y_max_entry_var, palette_label, window):
    """Update parameter display labels."""
    try:
        param_display_labels["Width"].config(text=width_entry_var.get())
        param_display_labels["Height"].config(text=height_entry_var.get())
        param_display_labels["Max Iterations"].config(text=max_iter_entry_var.get())
        param_display_labels["X Min"].config(text=x_min_entry_var.get())
        param_display_labels["X Max"].config(text=x_max_entry_var.get())
        param_display_labels["Y Min"].config(text=y_min_entry_var.get())
        param_display_labels["Y Max"].config(text=y_max_entry_var.get())
        
        # Get Palette name from the Palette label
        palette_text = palette_label.cget("text")
        if palette_text.startswith("Current: "):
            palette_name = palette_text[len("Current: "):]
            param_display_labels["Palette"].config(text=palette_name)
        
        # Update fractal type display
        fractal_type = get_fractal_type()
        halley_power = get_halley_power()
        if fractal_type == "Mandelbrot":
            param_display_labels["Fractal Type"].config(text="Mandelbrot")
        else:
            param_display_labels["Fractal Type"].config(text=f"Halley (n={halley_power})")
    except:
        pass
    
    # Schedule the next update
    window.after(500, lambda: update_parameter_display(param_display_labels, width_entry_var, height_entry_var, 
                                                      max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                                                      y_min_entry_var, y_max_entry_var, palette_label, window))


if __name__ == '__main__':
    window = tk.Tk()
    window.title("Fractal Explorer - Mandelbrot & Halley")

    custom_palette = []
    
    # Create necessary directories if they don't exist
    palette_dir = "Palette"  # Define palette_dir here
    for directory in [palette_dir, "Fractal Images", "Parameter Files"]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Default Palette file path is now in the Palette subfolder
    default_palette = "GOODEGA2.MAP"
    palette_file_path = os.path.join(palette_dir, default_palette)
    
    # If the default Palette doesn't exist in the Palette folder,
    # check if it exists in the current directory and move it
    if not os.path.exists(palette_file_path) and os.path.exists(f"./{default_palette}"):
        shutil.copy(f"./{default_palette}", palette_file_path)
        print(f"Moved {default_palette} to {palette_dir} folder")
    
    try:
        with open(palette_file_path, "r") as f:
            for line in f:
                rgb_str = line.strip().split(',')
                r = int(rgb_str[0])
                g = int(rgb_str[1])
                b = int(rgb_str[2])
                custom_palette.append((r, g, b))
       # print(f"Loaded default Palette: {default_palette} with {len(custom_palette)} colors")
    except FileNotFoundError:
        print(f"Error: Palette file not found at: {palette_file_path}.")
        # Create a default grayscale Palette instead of exiting
        for i in range(256):
            custom_palette.append((i, i, i))
        print("Created default grayscale Palette.")

    # Create main container frame
    main_frame = tk.Frame(window)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create left panel for displaying current parameters
    left_panel = tk.Frame(main_frame, padx=5, pady=10, borderwidth=1, relief=tk.RIDGE)
    left_panel.pack(side=tk.LEFT, fill=tk.Y)
    
    # Add title for the left panel
    left_panel_title = tk.Label(left_panel, text="Current Parameters", font=("Arial", 10, "bold"))
    left_panel_title.pack(pady=(0, 10))
    
    # Create labels for displaying current parameter values
    param_display_labels = {}
    param_names_display = ["Width", "Height", "Max Iterations", "X Min", "X Max", "Y Min", "Y Max", "Palette", "Fractal Type"]
    
    for param_name in param_names_display:
        frame = tk.Frame(left_panel)
        frame.pack(fill=tk.X, pady=2)
        
        name_label = tk.Label(frame, text=f"{param_name}:", width=12, anchor="w")
        name_label.pack(side=tk.LEFT)
        
        value_label = tk.Label(frame, text="--", width=15, anchor="w")
        value_label.pack(side=tk.LEFT)
        
        param_display_labels[param_name] = value_label
    
    # Add elapsed time label
    time_frame = tk.Frame(left_panel)
    time_frame.pack(fill=tk.X, pady=2)
    time_label = tk.Label(time_frame, text="Time:", width=12, anchor="w")
    time_label.pack(side=tk.LEFT)
    elapsed_time_label = tk.Label(time_frame, text="--", width=15, anchor="w")
    elapsed_time_label.pack(side=tk.LEFT)

    # Create a progress bar frame
    progress_frame = tk.Frame(left_panel)
    progress_frame.pack(fill=tk.X, pady=10)

    # Set a fixed width for the progress frame to prevent layout shifts
    progress_frame.pack_propagate(False)  # Prevent the frame from resizing based on contents
    progress_frame.configure(width=200, height=45)  # Set fixed dimensions

    progress_label = tk.Label(progress_frame, text="Progress:", anchor="w")
    progress_label.pack(fill=tk.X)
    
    progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=200)
    progress_bar.pack(fill=tk.X, padx=5)  # Add some padding to ensure the bar doesn't touch the edges
    
    # Create a new frame for the edit button
    edit_frame = tk.Frame(left_panel)
    edit_frame.pack(fill=tk.X, pady=5)
    
    # Add button to edit parameter file
    def edit_parameter_file():
        """Edit the currently loaded parameter file or open Parameters directory."""
        global loaded_parameter_file
        if loaded_parameter_file and os.path.exists(loaded_parameter_file):
            try:
                subprocess.Popen(['xed', loaded_parameter_file])
            except Exception as e:
                print(f"Error opening file with Xed: {e}")
        else:
            # Open the Parameter Files directory with file manager
            params_dir = os.path.abspath("Parameter Files")
            if not os.path.exists(params_dir):
                os.makedirs(params_dir)
            try:
                subprocess.Popen(['xdg-open', params_dir])
            except Exception as e:
                print(f"Error opening Parameters directory: {e}")
    
    edit_params_button = tk.Button(edit_frame, text="Edit Params", command=edit_parameter_file)
    edit_params_button.pack(fill=tk.X, padx=5, pady=2)
    
    # Set UI globals
    set_ui_globals(progress_bar, progress_label, elapsed_time_label)
    
    canvas_width = 1000
    canvas_height = 950
    
    # Create a frame to hold the canvas and scrollbars
    canvas_frame = tk.Frame(main_frame)
    canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Create horizontal and vertical scrollbars
    h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
    
    # Create the canvas with scrollbar configuration
    canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, 
                      background="black",
                      xscrollcommand=h_scrollbar.set,
                      yscrollcommand=v_scrollbar.set)
    
    # Configure the scrollbars to scroll the canvas
    h_scrollbar.config(command=canvas.xview)
    v_scrollbar.config(command=canvas.yview)
    
    # Pack the scrollbars and canvas
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    side_panel = tk.Frame(main_frame, padx=5, pady=10)
    side_panel.pack(side=tk.RIGHT, fill=tk.Y)

    params_frame = tk.Frame(side_panel)
    params_frame.pack(pady=10)

    labels_entries = []
    param_names = ["Width:", "Height:", "Max Iterations:", "X Min:", "X Max:", "Y Min:", "Y Max:"]
    default_values = ["1000", "950", "255", "-2.0", "1.0", "-1.5", "1.5"]
    entry_vars = []

    x_min_entry = None
    x_max_entry = None
    y_min_entry = None
    y_max_entry = None
    width_entry = None
    height_entry = None
    max_iter_entry = None

    for i, param_name in enumerate(param_names):
        label = tk.Label(params_frame, text=param_name, anchor="w")
        label.pack(fill=tk.X)
        entry_var = tk.StringVar(value=default_values[i])
        entry = tk.Entry(params_frame, textvariable=entry_var)
        entry.pack(fill=tk.X, pady=(0, 5))
        labels_entries.append((label, entry))
        entry_vars.append(entry_var)

        if param_name == "X Min:":
            x_min_entry = entry
        elif param_name == "X Max:":
            x_max_entry = entry
        elif param_name == "Y Min:":
            y_min_entry = entry
        elif param_name == "Y Max:":
            y_max_entry = entry
        elif param_name == "Width:":
            width_entry = entry
        elif param_name == "Height:":
            height_entry = entry
        elif param_name == "Max Iterations:":
            max_iter_entry = entry

    width_entry_var = entry_vars[0]
    height_entry_var = entry_vars[1]
    max_iter_entry_var = entry_vars[2]
    x_min_entry_var = entry_vars[3]
    x_max_entry_var = entry_vars[4]
    y_min_entry_var = entry_vars[5]
    y_max_entry_var = entry_vars[6]

    button_frame = tk.Frame(side_panel)
    button_frame.pack(pady=10)

    # Create a separate frame for Palette label (informational, not a button)
    palette_label_frame = tk.Frame(button_frame)
    palette_label_frame.pack(fill=tk.X, pady=2)
    
    # Add Palette label to show current Palette
    palette_label = tk.Label(palette_label_frame, text=f"Current: {default_palette}")
    palette_label.pack(pady=5)

    # Row 1: Start Drawing | End Drawing
    start_end_button_frame = tk.Frame(button_frame)
    start_end_button_frame.pack(fill=tk.X, pady=2)

    # Modify the start_drawing_thread function to update parameter display
    def start_drawing_with_update(*args, **kwargs):
        update_parameter_display(param_display_labels, width_entry_var, height_entry_var, 
                               max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                               y_min_entry_var, y_max_entry_var, palette_label, window)
        return start_drawing_thread(*args, **kwargs)
    
    start_button = tk.Button(start_end_button_frame, text="Start Drawing", command=lambda: start_drawing_with_update(
        canvas,
        width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
        custom_palette))
    start_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    end_button = tk.Button(start_end_button_frame, text="End Drawing", command=end_drawing)
    end_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # Row 2: Load Defaults | Load Palette
    defaults_palette_button_frame = tk.Frame(button_frame)
    defaults_palette_button_frame.pack(fill=tk.X, pady=2)

    load_defaults_button = tk.Button(defaults_palette_button_frame, text="Load Defaults", command=lambda: load_default_parameters(
        width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
        default_values, canvas, palette_label, custom_palette))
    load_defaults_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    load_palette_button = tk.Button(
        defaults_palette_button_frame, 
        text="Load Palette", 
        command=lambda: load_palette(
            None, canvas,
            width_entry_var, height_entry_var, max_iter_entry_var,
            x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
            palette_label, custom_palette
        )
    )
    load_palette_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # Row 3: Save Image | Load Image
    image_button_frame = tk.Frame(button_frame)
    image_button_frame.pack(fill=tk.X, pady=2)
    
    save_button = tk.Button(image_button_frame, text="Save Image", command=save_fractal_image)
    save_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    load_button = tk.Button(image_button_frame, text="Load Image", command=lambda: load_fractal_image(canvas))
    load_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # Row 4: Save Params | Load Params
    params_button_frame = tk.Frame(button_frame)
    params_button_frame.pack(fill=tk.X, pady=2)
    
    save_params_button = tk.Button(
        params_button_frame, 
        text="Save Params", 
        command=lambda: save_parameters_file(
            width_entry_var, height_entry_var, max_iter_entry_var,
            x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
            palette_label, get_fractal_type(), get_halley_power()
        )
    )
    save_params_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def load_params_and_redraw():
        """Load parameters and redraw if needed."""
        global loaded_parameter_file
        result = load_parameters_file(
            width_entry_var, height_entry_var, max_iter_entry_var,
            x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
            palette_label, custom_palette
        )
        if result:
            # Unpack the tuple (params, file_path)
            params, file_path = result
            loaded_parameter_file = file_path
            # Load the Palette if it's specified in the parameters
            palette_file = params.get("Palette")
            if palette_file and palette_label:
                load_palette(palette_file, canvas,
                           width_entry_var, height_entry_var, max_iter_entry_var,
                           x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                           palette_label, custom_palette)
            else:
                # Just redraw with the loaded parameters (fractal type already updated in load_parameters_file)
                start_drawing_with_update(
                    canvas,
                    width_entry_var, height_entry_var, max_iter_entry_var,
                    x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                    custom_palette
                )
    
    load_params_button = tk.Button(
        params_button_frame, 
        text="Load Params", 
        command=load_params_and_redraw
    )
    load_params_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    canvas.bind("<ButtonPress-1>", lambda event: start_zoom_rect(event, canvas))
    canvas.bind("<B1-Motion>",     lambda event: update_zoom_rect(event, canvas))
    canvas.bind("<ButtonRelease-1>", lambda event: finish_zoom_rect(event, canvas,
                                                                    x_min_entry, x_max_entry,
                                                                    y_min_entry, y_max_entry,
                                                                    width_entry_var, height_entry_var,
                                                                    max_iter_entry_var,
                                                                    x_min_entry_var, x_max_entry_var,
                                                                    y_min_entry_var, y_max_entry_var,
                                                                    custom_palette))
    
    # Bind right-click to show context menu
    canvas.bind("<Button-3>", lambda event: show_context_menu(
        event, canvas, width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, custom_palette))
    
    # Start the parameter display update loop
    update_parameter_display(param_display_labels, width_entry_var, height_entry_var, 
                            max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                            y_min_entry_var, y_max_entry_var, palette_label, window)

    # Center the window on screen
    window.update_idletasks()  # Make sure window size is updated
    
    # Get window and screen dimensions
    try:
        window_width = window.winfo_width()
        window_height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        # Only center if we got valid screen dimensions
        if screen_width > 1 and screen_height > 1:
            # Calculate center position
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        else:
            # Fallback: just set a default size
            window.geometry('1200x1100')
    except Exception as e:
        print(f"Warning: Could not center window: {e}")
        window.geometry('1200x1100')

    window.mainloop()
