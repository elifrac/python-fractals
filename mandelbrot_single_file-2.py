import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import numpy as np
import io
try:
    from PIL import Image, ImageTk
    HAVE_IMAGETK = True
except ImportError:
    from PIL import Image
    ImageTk = None
    HAVE_IMAGETK = False
import time
import threading
import json
import os
import random
from tkinter import ttk

drawing_running = False
zoom_rectangle_id = None
zoom_start_x, zoom_start_y = None, None
current_tk_image = None         # To store the currently displayed ImageTk.PhotoImage
current_pil_image = None        # To store the currently generated PIL Image

# Global variables for fractal type and parameters
current_fractal_type = "Mandelbrot"  # Default fractal type
current_halley_power = 3  # Default power for Halley's fractal
context_menu = None  # Global variable to track the current context menu


def pil_image_to_tk(image):
    """
    Convert a PIL Image to a Tkinter PhotoImage.
    Uses ImageTk when available, otherwise falls back to tk.PhotoImage via in-memory PNG.
    """
    if HAVE_IMAGETK and ImageTk is not None:
        return ImageTk.PhotoImage(image=image)
    # Fallback: encode image as PNG in memory and feed to tk.PhotoImage
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
    return tk.PhotoImage(data=data)

def mandelbrot_set_iterations_pixel_yield_numpy(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Optimized Mandelbrot set calculation using NumPy vectorization.
    Returns a 2D NumPy array of iteration counts.
    """
    # Update progress
    progress_bar['value'] = 0
    progress_label.config(text="Progress: Initializing...")
    
    # Create coordinate arrays with correct orientation
    real = np.linspace(x_min, x_max, width, dtype=np.float64)
    imag = np.linspace(y_max, y_min, height, dtype=np.float64)
    
    # Create meshgrid for vectorized computation
    real_grid, imag_grid = np.meshgrid(real, imag)
    c = real_grid + 1j * imag_grid
    
    # Initialize arrays
    z = np.zeros_like(c)
    iterations = np.zeros(c.shape, dtype=int)
    mask = np.ones(c.shape, dtype=bool)
    
    # Perform iteration
    for i in range(max_iter):
        if not drawing_running:  # Check if we should stop
            return iterations
            
        z[mask] = z[mask]**2 + c[mask]
        escaped = np.abs(z) > 2
        iterations[escaped & mask] = i + 1
        mask &= ~escaped
        
        # Update progress every few iterations
        if i % 5 == 0:
            progress = int(25 * (i / max_iter))  # Use first 25% for calculation
            progress_bar['value'] = progress
            progress_label.config(text=f"Progress: Calculating ({i}/{max_iter} iterations)")
            progress_bar.update_idletasks()
        
        if not np.any(mask):
            break
    
    return iterations

def halley_fractal_iterations(width, height, max_iter, x_min, x_max, y_min, y_max, n=3):
    """
    Compute Halley's fractal iterations using NumPy vectorization.
    Returns iteration counts and root indices.
    """
    # Update progress
    progress_bar['value'] = 0
    progress_label.config(text="Progress: Initializing...")
    progress_bar.update_idletasks()
    
    # Create coordinate arrays
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_max, y_min, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize complex plane
    c = X + 1j * Y
    z = np.copy(c)
    
    # Initialize arrays for iteration counts and root indices
    iterations = np.zeros((height, width), dtype=np.int32)
    root_indices = np.zeros((height, width), dtype=np.int32)
    
    # Create array of nth roots of unity for convergence checking
    roots_of_unity = np.exp(2j * np.pi * np.arange(n) / n)
    
    # Tolerance for convergence
    tolerance = 1e-6
    
    # Mask for points still being iterated
    mask = np.ones((height, width), dtype=bool)
    
    # Iterate
    for i in range(max_iter):
        if not drawing_running:  # Check if we should stop
            return iterations, root_indices
            
        if not np.any(mask):
            break
            
        # Compute z^n - 1 (the function)
        z_n = np.power(z, n)
        f_z = z_n - 1
        
        # Compute the first derivative: n * z^(n-1)
        df_z = n * np.power(z, n-1)
        
        # Compute the second derivative: n * (n-1) * z^(n-2)
        d2f_z = n * (n-1) * np.power(z, n-2)
        
        # Halley's method formula
        # z_new = z - (2 * f_z * df_z) / (2 * df_z^2 - f_z * d2f_z)
        numerator = 2 * f_z * df_z
        denominator = 2 * df_z * df_z - f_z * d2f_z
        
        # Avoid division by zero
        safe_denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        
        # Update z using Halley's method
        z_new = z - numerator / safe_denominator
        
        # Check for convergence to any root of unity
        for j, root in enumerate(roots_of_unity):
            # Calculate distance to this root
            distance = np.abs(z_new - root)
            
            # Points that converged to this root
            converged = (distance < tolerance) & mask
            
            if np.any(converged):
                # Record iteration count and root index
                iterations[converged] = i + 1
                root_indices[converged] = j + 1  # +1 so that 0 can represent non-converged points
                
                # Remove converged points from mask
                mask[converged] = False
        
        # Check for divergence (points that are clearly not converging)
        diverged = (np.abs(z_new) > 1e6) & mask
        if np.any(diverged):
            iterations[diverged] = i + 1
            mask[diverged] = False
        
        # Update z for next iteration (only for points still in the mask)
        z[mask] = z_new[mask]
        
        # Update progress every few iterations
        if i % 5 == 0:
            progress = int(25 * (i / max_iter))  # Use first 25% for calculation
            progress_bar['value'] = progress
            progress_label.config(text=f"Progress: Calculating ({i}/{max_iter} iterations)")
            progress_bar.update_idletasks()
    
    return iterations, root_indices

def draw_mandelbrot_scanlines_tkinter_palette(canvas, width, height, max_iter, x_min, x_max, y_min, y_max, color_palette, scanlines_update=950):
    global drawing_running, current_tk_image, current_pil_image
    
    start_time = time.time()  # Start timing
    
    # Calculate all iteration counts at once
    iteration_counts = mandelbrot_set_iterations_pixel_yield_numpy(width, height, max_iter, x_min, x_max, y_min, y_max)
    
    if not drawing_running:  # Check if we should stop
        return
    
    # Update progress for coloring
    progress_bar['value'] = 25
    progress_label.config(text="Progress: Coloring...")
    progress_bar.update_idletasks()
    
    if not drawing_running:  # Check if we should stop
        return
    
    # Create RGB array for the image
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill black for non-escaped points (iteration_count == 0)
    img_array[iteration_counts == 0] = [0, 0, 0]
    
    # Vectorized coloring: map iteration counts to palette colors
    # Create a mask for escaped points (iteration_count > 0)
    escaped_mask = iteration_counts > 0
    
    if np.any(escaped_mask):
        # Get palette indices using modulo for cycling through palette
        palette_size = len(color_palette)
        color_indices = (iteration_counts[escaped_mask] - 1) % palette_size
        
        # Convert palette to numpy array for efficient indexing
        palette_array = np.array(color_palette, dtype=np.uint8)
        
        # Assign colors using advanced indexing (much faster than loop)
        img_array[escaped_mask] = palette_array[color_indices]
    
    # Update progress for final rendering
    progress_bar['value'] = 75
    progress_label.config(text="Progress: Rendering...")
    progress_bar.update_idletasks()
    
    # Convert NumPy array to PIL Image
    image = Image.fromarray(img_array)
    current_pil_image = image
    
    # Display the image
    if drawing_running:
        tk_image = pil_image_to_tk(image)
        current_tk_image = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.tk_image = tk_image
        canvas.config(scrollregion=(0, 0, width, height))
        canvas.update()
        
        # Complete progress
        progress_bar['value'] = 100
        progress_label.config(text="Progress: Complete")
        progress_bar.update_idletasks()
        
        # Calculate and display elapsed time
        elapsed_time = time.time() - start_time
        elapsed_time_label.config(text=f"{elapsed_time:.2f} sec")

def draw_halley_fractal(canvas, width, height, max_iter, x_min, x_max, y_min, y_max, color_palette, n=3):
    """Draw Halley's fractal on the canvas"""
    global drawing_running, current_tk_image, current_pil_image
    
    start_time = time.time()  # Start timing
    
    # Reset progress bar
    progress_bar['value'] = 0
    progress_label.config(text="Progress: Calculating...")
    canvas.update_idletasks()
    
    # Compute iterations and root indices
    iterations, root_indices = halley_fractal_iterations(width, height, max_iter, x_min, x_max, y_min, y_max, n)
    
    # Update progress
    progress_bar['value'] = 50
    progress_label.config(text="Progress: Coloring...")
    canvas.update_idletasks()
    
    # Create an image from the iteration counts
    img = Image.new('RGB', (width, height), color='black')
    pixels = img.load()
    
    # Fill the image
    for y in range(height):
        for x in range(width):
            iter_count = iterations[y, x]
            root_idx = root_indices[y, x]
            
            # Points that converged to a root of unity have root_idx >= 1
            if root_idx > n:
                # Use the custom Palette based on root index
                color_idx = (root_idx - 1) % len(color_palette)
                pixels[x, y] = color_palette[color_idx]
            else:  # Point did not converge
                # Use iteration count for non-converged points
                color_idx = iter_count % len(color_palette)
                pixels[x, y] = color_palette[color_idx]
    
    # Update progress
    progress_bar['value'] = 75
    progress_label.config(text="Progress: Rendering...")
    canvas.update_idletasks()
    
    # Convert the image to PhotoImage and display on canvas
    current_pil_image = img
    
    # Display the image
    if drawing_running:
        tk_image = pil_image_to_tk(img)
        current_tk_image = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.tk_image = tk_image
        canvas.config(scrollregion=(0, 0, width, height))
        canvas.update()
        
        # Complete progress
        progress_bar['value'] = 100
        progress_label.config(text="Progress: Complete")
        canvas.update_idletasks()
        
        # Calculate and display elapsed time
        elapsed_time = time.time() - start_time
        elapsed_time_label.config(text=f"{elapsed_time:.2f} sec")
    
    return iterations, root_indices

def start_drawing_thread(canvas, width_entry_var, height_entry_var, max_iter_entry_var, x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, color_palette):
    global drawing_running, current_fractal_type, current_halley_power
    drawing_running = True
    try:
        # Reset elapsed time display
        elapsed_time_label.config(text="--")
        
        # Reset progress bar at start
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
        drawing_running = False
        # Reset progress bar on error
        progress_bar['value'] = 0
        progress_label.config(text="Progress: Error")
        canvas.update_idletasks()

def end_drawing():
    global drawing_running
    drawing_running = False
    # Reset progress bar
    progress_bar['value'] = 0
    progress_label.config(text="Progress: Cancelled")

def start_zoom_rect(event, canvas):
    global zoom_start_x, zoom_start_y, zoom_rectangle_id
    # Hide context menu when starting zoom
    hide_context_menu()
    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    zoom_start_x, zoom_start_y = canvas_x, canvas_y
    zoom_rectangle_id = canvas.create_rectangle(zoom_start_x, zoom_start_y, zoom_start_x, zoom_start_y, outline="white")

def update_zoom_rect(event, canvas):
    global zoom_rectangle_id
    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    canvas.coords(zoom_rectangle_id, zoom_start_x, zoom_start_y, canvas_x, canvas_y)

def finish_zoom_rect(event, canvas, x_min_entry_widget, x_max_entry_widget, y_min_entry_widget, y_max_entry_widget, width_entry_var, height_entry_var, max_iter_entry_var, color_palette):
    global zoom_rectangle_id, drawing_running, custom_palette
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

    # Update entry widgets
    x_min_entry_widget.delete(0, tk.END)
    x_min_entry_widget.insert(0, f"{new_x_min:.12f}")

    x_max_entry_widget.delete(0, tk.END)
    x_max_entry_widget.insert(0, f"{new_x_max:.12f}")

    y_min_entry_widget.delete(0, tk.END)
    y_min_entry_widget.insert(0, f"{new_y_min:.12f}")

    y_max_entry_widget.delete(0, tk.END)
    y_max_entry_widget.insert(0, f"{new_y_max:.12f}")

    # Draw the new zoomed view
    start_drawing_thread(
        canvas,
        width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
        custom_palette  # Use custom_palette instead of color_palette
    )

def load_default_parameters(width_entry_var, height_entry_var, max_iter_entry_var, x_min_entry_var, 
                         x_max_entry_var, y_min_entry_var, y_max_entry_var, 
                         default_values, canvas=None, palette_label=None):
    # Set default parameter values
    width_entry_var.set(default_values[0])
    height_entry_var.set(default_values[1])
    max_iter_entry_var.set(default_values[2])
    x_min_entry_var.set(default_values[3])
    x_max_entry_var.set(default_values[4])
    y_min_entry_var.set(default_values[5])
    y_max_entry_var.set(default_values[6])
    
    # Load default Palette if canvas and palette_label are provided
    if canvas and palette_label:
        # Load the default Palette (GOODEGA2.MAP)
        load_palette("GOODEGA2.MAP", canvas, width_entry_var, height_entry_var, max_iter_entry_var,
                    x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                    palette_label)

def save_fractal_image(canvas): # Removed width_entry_var, height_entry_var - not needed
    """
    Saves the current Mandelbrot image displayed on the canvas.
    Now saves directly from the stored PIL Image, without re-rendering.
    """
    global current_pil_image # Use the global PIL Image
    
    # Create Fractal Images directory if it doesn't exist
    images_dir = "Fractal Images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    if current_pil_image: # Check if there is a PIL image to save
        file_path = filedialog.asksaveasfilename(
            initialdir=images_dir,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Determine file format based on extension
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext in ['.jpg', '.jpeg']:
                    # For JPEG, convert to RGB if needed (JPEG doesn't support alpha channel)
                    if current_pil_image.mode in ['RGBA', 'P']:
                        rgb_image = current_pil_image.convert('RGB')
                        rgb_image.save(file_path, quality=95)  # Higher quality for JPEG
                    else:
                        current_pil_image.save(file_path, quality=95)
                else:
                    # For PNG and other formats, save as is
                    current_pil_image.save(file_path)
                    
                print(f"Image saved to {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
    else:
        print("No image to save. Draw the Mandelbrot set first.")


def load_fractal_image(canvas):
    """
    Loads a Mandelbrot image from a file and displays it on the canvas.
    """
    # Create Fractal Images directory if it doesn't exist
    images_dir = "Fractal Images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    file_path = filedialog.askopenfilename(
        initialdir=images_dir,
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("PNG files", "*.png"), 
                  ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
    )
    if file_path:
        try:
            image = Image.open(file_path)
            width, height = image.size
            tk_image = pil_image_to_tk(image)
            global current_tk_image, current_pil_image # Update global PIL image too
            current_tk_image = tk_image
            current_pil_image = image # Store the loaded PIL Image
            canvas.config(width=width, height=height)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            canvas.tk_image = tk_image
            # Configure the scrollregion to match the image size
            canvas.config(scrollregion=(0, 0, width, height))
            print(f"Image loaded from {file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

def save_parameters_file(width_entry_var, height_entry_var, max_iter_entry_var, 
                           x_min_entry_var, x_max_entry_var, y_min_entry_var, 
                           y_max_entry_var, palette_label=None):
    global current_fractal_type, current_halley_power
    
    # Get the current Palette name from the label
    current_palette = "GOODEGA2.MAP"  # Default value
    if palette_label:
        label_text = palette_label.cget("text")
        if label_text.startswith("Current: "):
            current_palette = label_text[len("Current: "):]
    
    params = {
        "width": width_entry_var.get(),
        "height": height_entry_var.get(),
        "max_iter": max_iter_entry_var.get(),
        "x_min": x_min_entry_var.get(),
        "x_max": x_max_entry_var.get(),
        "y_min": y_min_entry_var.get(),
        "y_max": y_max_entry_var.get(),
        "Palette": current_palette,  # Save the current Palette name
        "fractal_type": current_fractal_type,  # Save the fractal type
        "halley_power": current_halley_power   # Save the Halley power
    }
    
    # Create Parameter Files directory if it doesn't exist
    params_dir = "Parameter Files"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
        
    file_path = filedialog.asksaveasfilename(
        initialdir=params_dir,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if file_path:
        try:
            with open(file_path, "w") as fp:
                json.dump(params, fp, indent=4)
            print(f"Parameters saved to {file_path}")
        except Exception as e:
            print(f"Error saving parameters: {e}")


def load_parameters_file(width_entry_var, height_entry_var, max_iter_entry_var, 
                           x_min_entry_var, x_max_entry_var, y_min_entry_var, 
                           y_max_entry_var, canvas, color_palette, palette_label=None):
    global current_fractal_type, current_halley_power, custom_palette
    
    # Create Parameter Files directory if it doesn't exist
    params_dir = "Parameter Files"
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
        
    file_path = filedialog.askopenfilename(
        initialdir=params_dir,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
    )
    if file_path:
        try:
            with open(file_path, "r") as fp:
                params = json.load(fp)
            width_entry_var.set(params.get("width", width_entry_var.get()))
            height_entry_var.set(params.get("height", height_entry_var.get()))
            max_iter_entry_var.set(params.get("max_iter", max_iter_entry_var.get()))
            x_min_entry_var.set(params.get("x_min", x_min_entry_var.get()))
            x_max_entry_var.set(params.get("x_max", x_max_entry_var.get()))
            y_min_entry_var.set(params.get("y_min", y_min_entry_var.get()))
            y_max_entry_var.set(params.get("y_max", y_max_entry_var.get()))
            
            # Load fractal type and power if available
            if "fractal_type" in params:
                current_fractal_type = params["fractal_type"]
            if "halley_power" in params:
                current_halley_power = params["halley_power"]
                
            print(f"Parameters loaded from {file_path}")
            
            # Load the Palette if it's specified in the parameters
            palette_file = params.get("Palette")
            if palette_file and palette_label:
                load_palette(palette_file, canvas, width_entry_var, height_entry_var, max_iter_entry_var,
                           x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                           palette_label)
            else:
                # Start drawing immediately after parameters are loaded
                start_drawing_thread(
                    canvas,
                    width_entry_var, height_entry_var, max_iter_entry_var,
                    x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                    custom_palette  # Use custom_palette instead of color_palette
                )
        except Exception as e:
            print(f"Error loading parameters: {e}")

def load_palette(palette_file=None, canvas=None, width_entry_var=None, height_entry_var=None, 
                max_iter_entry_var=None, x_min_entry_var=None, x_max_entry_var=None, y_min_entry_var=None, 
                y_max_entry_var=None, palette_label=None):
    """Load a color Palette from a file"""
    global custom_palette
    
    try:
        palette_dir = "Palette"
        
        if palette_file is None:
            # Open file dialog to select Palette
            filetypes = [("Palette files", "*.MAP"), ("All files", "*.*")]
            palette_file = filedialog.askopenfilename(
                initialdir=palette_dir,
                title="Select Palette File",
                filetypes=filetypes
            )
            
            # If user cancels the dialog
            if not palette_file:
                return
        else:
            # For default Palette loading
            palette_file = os.path.join(palette_dir, palette_file)
        
        # Extract just the filename for display
        palette_name = os.path.basename(palette_file)
        if palette_label:
            palette_label.config(text=f"Current: {palette_name}")
        
        # Clear the current Palette
        custom_palette.clear()
        
        # Load the Palette file
        try:
            # First try to load as a text file with RGB values
            with open(palette_file, "r") as f:
                for line in f:
                    rgb_str = line.strip().split(',')
                    r = int(rgb_str[0])
                    g = int(rgb_str[1])
                    b = int(rgb_str[2])
                    custom_palette.append((r, g, b))
        except:
            # If that fails, try to load as a binary file
            with open(palette_file, 'rb') as f:
                palette_data = f.read()
                
                # Process the Palette data - assuming 3 bytes per color (RGB)
                for i in range(0, len(palette_data), 3):
                    if i+2 < len(palette_data):
                        r, g, b = palette_data[i], palette_data[i+1], palette_data[i+2]
                        custom_palette.append((r, g, b))
        
        # print(f"Loaded Palette: {palette_name} with {len(custom_palette)} colors")
        
        # Redraw the fractal with the new Palette if canvas is provided
        if canvas and all(var is not None for var in [width_entry_var, height_entry_var, max_iter_entry_var, 
                                                     x_min_entry_var, x_max_entry_var, y_min_entry_var, 
                                                     y_max_entry_var]):
            start_drawing_thread(
                canvas,
                width_entry_var, height_entry_var, max_iter_entry_var,
                x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                custom_palette
            )
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Palette: {str(e)}")
        print(f"Error loading Palette: {e}")

def show_context_menu(event, canvas, width_entry_var, height_entry_var, max_iter_entry_var, 
                     x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, color_palette):
    """Show context menu on right-click"""
    global current_fractal_type, current_halley_power, custom_palette, context_menu
    
    # Destroy previous context menu if it exists
    if context_menu is not None:
        try:
            context_menu.destroy()
        except tk.TclError:
            pass
    
    # Get the canvas coordinates, accounting for scroll position
    canvas_x = canvas.canvasx(event.x)
    canvas_y = canvas.canvasy(event.y)
    
    context_menu = Menu(canvas, tearoff=0)
    
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
    
    # Display the menu at the click position
    context_menu.post(event.x_root, event.y_root)

def hide_context_menu():
    """Hide the context menu"""
    global context_menu
    if context_menu is not None:
        try:
            context_menu.unpost()
        except tk.TclError:
            pass

def change_fractal_type(fractal_type, canvas, width_entry_var, height_entry_var, max_iter_entry_var, 
                       x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, color_palette, power=3):
    """Change the fractal type and redraw"""
    global current_fractal_type, current_halley_power, custom_palette
    
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
        custom_palette  # Use custom_palette instead of color_palette
    )

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
        import shutil
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
        print(f"Loaded default Palette: {default_palette} with {len(custom_palette)} colors")
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
    
    # Function to update parameter display
    def update_parameter_display():
        global current_fractal_type, current_halley_power
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
            if current_fractal_type == "Mandelbrot":
                param_display_labels["Fractal Type"].config(text="Mandelbrot")
            else:
                param_display_labels["Fractal Type"].config(text=f"Halley (n={current_halley_power})")
        except:
            pass
        
        # Schedule the next update
        window.after(500, update_parameter_display)
    
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
    original_start_drawing = start_drawing_thread
    def start_drawing_with_update(*args, **kwargs):
        update_parameter_display()  # Update parameter display when drawing starts
        return original_start_drawing(*args, **kwargs)
    
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
        default_values, canvas, palette_label))
    load_defaults_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    load_palette_button = tk.Button(
        defaults_palette_button_frame, 
        text="Load Palette", 
        command=lambda: load_palette(
            None, canvas,
            width_entry_var, height_entry_var, max_iter_entry_var,
            x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
            palette_label
        )
    )
    load_palette_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # Row 3: Save Image | Load Image
    image_button_frame = tk.Frame(button_frame)
    image_button_frame.pack(fill=tk.X, pady=2)
    
    save_button = tk.Button(image_button_frame, text="Save Image", command=lambda: save_fractal_image(canvas))
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
            palette_label
        )
    )
    save_params_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    load_params_button = tk.Button(
        params_button_frame, 
        text="Load Params", 
        command=lambda: load_parameters_file(
            width_entry_var, height_entry_var, max_iter_entry_var,
            x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
            canvas, custom_palette, palette_label
        )
    )
    load_params_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    canvas.bind("<ButtonPress-1>", lambda event: start_zoom_rect(event, canvas))
    canvas.bind("<B1-Motion>",     lambda event: update_zoom_rect(event, canvas))
    canvas.bind("<ButtonRelease-1>", lambda event: finish_zoom_rect(event, canvas,
                                                                    x_min_entry, x_max_entry,
                                                                    y_min_entry, y_max_entry,
                                                                    width_entry_var, height_entry_var,
                                                                    max_iter_entry_var,
                                                                    custom_palette))
    
    # Bind right-click to show context menu
    canvas.bind("<Button-3>", lambda event: show_context_menu(
        event, canvas, width_entry_var, height_entry_var, max_iter_entry_var,
        x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var, custom_palette))
    
    # Start the parameter display update loop
    update_parameter_display()

    # Center the window on screen
    window.update_idletasks()  # Make sure window size is updated
    window.withdraw()  # Hide the window temporarily
    # Get window and screen dimensions
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    # Calculate center position
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    # Position the window and show it
    window.geometry(f'{window_width}x{window_height}+{x}+{y}')
    window.deiconify()  # Show the window again

    window.mainloop()