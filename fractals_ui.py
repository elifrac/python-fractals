"""
UI components and event handlers for fractal explorer.
"""
import tkinter as tk
from tkinter import Menu
import threading
import fractals_core
from fractals_core import set_progress_controls
from fractals_render import (
    draw_mandelbrot_scanlines_tkinter_palette,
    draw_halley_fractal,
    set_elapsed_time_label
)
from fractals_io import load_palette

# Global state
zoom_rectangle_id = None
zoom_start_x, zoom_start_y = None, None
current_fractal_type = "Mandelbrot"  # Default fractal type
current_halley_power = 3  # Default power for Halley's fractal
progress_bar = None
progress_label = None
elapsed_time_label = None
current_context_menu = None  # Track the current context menu


def set_ui_globals(progress_bar_ref, progress_label_ref, elapsed_time_label_ref):
    """Set references to UI components."""
    global progress_bar, progress_label, elapsed_time_label
    progress_bar = progress_bar_ref
    progress_label = progress_label_ref
    elapsed_time_label = elapsed_time_label_ref
    set_progress_controls(progress_bar_ref, progress_label_ref)
    set_elapsed_time_label(elapsed_time_label_ref)


def get_fractal_type():
    """Get current fractal type."""
    return current_fractal_type


def get_halley_power():
    """Get current Halley power."""
    return current_halley_power


def start_drawing_thread(canvas, width_entry_var, height_entry_var, max_iter_entry_var, 
                        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, color_palette):
    """Start drawing fractal in a separate thread."""
    global current_fractal_type, current_halley_power, progress_bar, progress_label, elapsed_time_label
    
    fractals_core.drawing_running = True
    try:
        # Reset elapsed time display
        if elapsed_time_label:
            elapsed_time_label.config(text="--")
        
        # Reset progress bar at start
        if progress_bar and progress_label:
            progress_bar['value'] = 0
            progress_label.config(text="Progress: Starting...")
            canvas.update_idletasks()
        
        width = int(width_entry_var.get())
        height = int(height_entry_var.get())
        max_iter = int(max_iter_entry_var.get())
        x_min = float(x_min_entry_var.get())
        x_max = float(x_max_entry_var.get())
        y_min = float(y_min_entry_var.get())
        y_max = float(y_max_entry_var.get())

        canvas.config(width=width, height=height)
        canvas.delete("all")
        # Reset the scrollregion when starting a new drawing
        canvas.config(scrollregion=(0, 0, width, height))
        # Reset the scroll position to the top-left corner
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        if current_fractal_type == "Mandelbrot":
            threading.Thread(target=draw_mandelbrot_scanlines_tkinter_palette, args=(
                canvas, width, height, max_iter, x_min, x_max, y_min, y_max, color_palette, 950)).start()
        else:  # Halley's fractal
            threading.Thread(target=draw_halley_fractal, args=(
                canvas, width, height, max_iter, x_min, x_max, y_min, y_max, color_palette, current_halley_power)).start()

    except ValueError:
        print("Error: Invalid input in parameter fields. Please enter integers and floats correctly.")
        fractals_core.drawing_running = False
        # Reset progress bar on error
        if progress_bar and progress_label:
            progress_bar['value'] = 0
            progress_label.config(text="Progress: Error")
            canvas.update_idletasks()


def end_drawing():
    """Stop the current drawing operation."""
    fractals_core.drawing_running = False
    # Reset progress bar
    if progress_bar and progress_label:
        progress_bar['value'] = 0
        progress_label.config(text="Progress: Cancelled")


def dismiss_context_menu():
    """Dismiss the current context menu if it exists."""
    global current_context_menu
    if current_context_menu:
        try:
            current_context_menu.destroy()
            current_context_menu = None
        except:
            current_context_menu = None


def start_zoom_rect(event, canvas):
    """Start drawing zoom rectangle."""
    global zoom_start_x, zoom_start_y, zoom_rectangle_id
    # Dismiss context menu if it's open
    dismiss_context_menu()
    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    zoom_start_x, zoom_start_y = canvas_x, canvas_y
    zoom_rectangle_id = canvas.create_rectangle(zoom_start_x, zoom_start_y, zoom_start_x, zoom_start_y, outline="white")


def update_zoom_rect(event, canvas):
    """Update zoom rectangle during drag."""
    global zoom_rectangle_id
    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    canvas.coords(zoom_rectangle_id, zoom_start_x, zoom_start_y, canvas_x, canvas_y)


def finish_zoom_rect(event, canvas, x_min_entry_widget, x_max_entry_widget, y_min_entry_widget, y_max_entry_widget, 
                     width_entry_var, height_entry_var, max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                     y_min_entry_var, y_max_entry_var, custom_palette):
    """Finish zoom rectangle and update view."""
    global zoom_rectangle_id
    if zoom_rectangle_id:
        canvas.delete(zoom_rectangle_id)

    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    x_end, y_end = canvas_x, canvas_y
    x1 = min(zoom_start_x, x_end)
    y1 = min(zoom_start_y, y_end)
    x2 = max(zoom_start_x, x_end)
    y2 = max(zoom_start_y, y_end)

    if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
        return

    canvas_width = int(width_entry_var.get())
    canvas_height = int(height_entry_var.get())
    
    # Get current complex plane boundaries
    x_min_current = float(x_min_entry_widget.get())
    x_max_current = float(x_max_entry_widget.get())
    y_min_current = float(y_min_entry_widget.get())
    y_max_current = float(y_max_entry_widget.get())
    
    x_range = x_max_current - x_min_current
    y_range = y_max_current - y_min_current
    
    # Map pixel coordinates to complex plane coordinates
    new_x_min = x_min_current + (x1 / canvas_width) * x_range
    new_x_max = x_min_current + (x2 / canvas_width) * x_range
    
    # For y-coordinates, remember that y increases downward in pixel space
    # but typically increases upward in the complex plane
    new_y_max = y_max_current - (y1 / canvas_height) * y_range
    new_y_min = y_max_current - (y2 / canvas_height) * y_range
    
    # Preserve aspect ratio
    x_center = (new_x_min + new_x_max) / 2
    y_center = (new_y_min + new_y_max) / 2
    
    new_x_range = new_x_max - new_x_min
    new_y_range = new_y_max - new_y_min
    
    aspect_ratio = canvas_width / canvas_height
    if new_x_range / new_y_range > aspect_ratio:
        # Too wide, adjust height
        new_y_range = new_x_range / aspect_ratio
    else:
        # Too tall, adjust width
        new_x_range = new_y_range * aspect_ratio
    
    new_x_min = x_center - new_x_range / 2
    new_x_max = x_center + new_x_range / 2
    new_y_min = y_center - new_y_range / 2
    new_y_max = y_center + new_y_range / 2

    # Update entry widgets AND StringVar variables
    x_min_entry_widget.delete(0, tk.END)
    x_min_entry_widget.insert(0, f"{new_x_min:.12f}")
    x_min_entry_var.set(f"{new_x_min:.12f}")

    x_max_entry_widget.delete(0, tk.END)
    x_max_entry_widget.insert(0, f"{new_x_max:.12f}")
    x_max_entry_var.set(f"{new_x_max:.12f}")

    y_min_entry_widget.delete(0, tk.END)
    y_min_entry_widget.insert(0, f"{new_y_min:.12f}")
    y_min_entry_var.set(f"{new_y_min:.12f}")

    y_max_entry_widget.delete(0, tk.END)
    y_max_entry_widget.insert(0, f"{new_y_max:.12f}")
    y_max_entry_var.set(f"{new_y_max:.12f}")

    # Draw the new zoomed view
    start_drawing_thread(
        canvas,
        width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
        custom_palette
    )


def load_default_parameters(width_entry_var, height_entry_var, max_iter_entry_var, x_min_entry_var, 
                         x_max_entry_var, y_min_entry_var, y_max_entry_var, 
                         default_values, canvas=None, palette_label=None, custom_palette=None):
    """Load default parameter values."""
    # Set default parameter values
    width_entry_var.set(default_values[0])
    height_entry_var.set(default_values[1])
    max_iter_entry_var.set(default_values[2])
    x_min_entry_var.set(default_values[3])
    x_max_entry_var.set(default_values[4])
    y_min_entry_var.set(default_values[5])
    y_max_entry_var.set(default_values[6])
    
    # Load default Palette if canvas and palette_label are provided
    if canvas and palette_label and custom_palette:
        # Load the default Palette (GOODEGA2.MAP)
        load_palette("GOODEGA2.MAP", canvas, width_entry_var, height_entry_var, max_iter_entry_var,
                    x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                    palette_label, custom_palette)


def show_context_menu(event, canvas, width_entry_var, height_entry_var, max_iter_entry_var, 
                     x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, custom_palette):
    """Show context menu on right-click"""
    global current_fractal_type, current_halley_power, current_context_menu
    
    # Destroy any existing context menu first
    if current_context_menu:
        try:
            current_context_menu.destroy()
        except:
            pass
    
    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    
    context_menu = Menu(canvas, tearoff=0)
    current_context_menu = context_menu  # Store reference to current menu
    
    # Add Mandelbrot option
    context_menu.add_checkbutton(
        label="Mandelbrot", 
        command=lambda: change_fractal_type("Mandelbrot", canvas, width_entry_var, height_entry_var, 
                                           max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                                           y_min_entry_var, y_max_entry_var, custom_palette),
        onvalue=1,
        offvalue=0,
        variable=tk.IntVar(value=1 if current_fractal_type == "Mandelbrot" else 0)
    )
    
    # Add Halley's Map submenu
    halley_menu = Menu(context_menu, tearoff=0)
    
    # Add different powers for Halley's method
    for power in range(2, 11):  # Powers from 2 to 10
        halley_menu.add_checkbutton(
            label=f"Power {power}",
            command=lambda p=power: change_fractal_type("Halley", canvas, width_entry_var, height_entry_var, 
                                                      max_iter_entry_var, x_min_entry_var, x_max_entry_var, 
                                                      y_min_entry_var, y_max_entry_var, custom_palette, p),
            onvalue=1,
            offvalue=0,
            variable=tk.IntVar(value=1 if current_fractal_type == "Halley" and current_halley_power == power else 0)
        )
    
    context_menu.add_cascade(label="Halley's Map", menu=halley_menu)
    
    # Add a callback to clear the menu reference when it's dismissed
    def on_menu_dismiss():
        global current_context_menu
        current_context_menu = None
    
    # Bind to menu dismissal (when user clicks elsewhere or selects an item)
    context_menu.bind("<Unmap>", lambda e: on_menu_dismiss())
    
    # Display the menu at the click position
    context_menu.post(event.x_root, event.y_root)


def change_fractal_type(fractal_type, canvas, width_entry_var, height_entry_var, max_iter_entry_var, 
                       x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, custom_palette, power=3):
    """Change the fractal type and redraw"""
    global current_fractal_type, current_halley_power
    
    # Update global variables
    current_fractal_type = fractal_type
    if fractal_type == "Halley":
        current_halley_power = power
    
    # Set default view parameters based on fractal type
    if fractal_type == "Mandelbrot":
        # Default Mandelbrot view
        x_min_entry_var.set("-2.0")
        x_max_entry_var.set("1.0")
        y_min_entry_var.set("-1.5")
        y_max_entry_var.set("1.5")
    else:  # Halley's fractal
        # Default Halley view (centered at origin with good zoom level)
        x_min_entry_var.set("-1.5")
        x_max_entry_var.set("1.5")
        y_min_entry_var.set("-1.5")
        y_max_entry_var.set("1.5")
        
        # Increase max iterations for better detail
        max_iter_entry_var.set("100")
    
    # Redraw the fractal
    start_drawing_thread(
        canvas,
        width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
        custom_palette
    )
