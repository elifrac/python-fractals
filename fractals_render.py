"""
Fractal rendering functions that create and display images.
"""
import tkinter as tk
import time
import numpy as np
from PIL import Image
from fractals_utils import pil_image_to_tk
import fractals_core
from fractals_core import (
    mandelbrot_set_iterations_pixel_yield_numpy,
    halley_fractal_iterations
)

# Global state for images
current_tk_image = None
current_pil_image = None
elapsed_time_label = None


def set_elapsed_time_label(label_ref):
    """Set reference to elapsed time label for updates."""
    global elapsed_time_label
    elapsed_time_label = label_ref


def get_current_pil_image():
    """Get the current PIL image."""
    return current_pil_image


def set_current_pil_image(image):
    """Set the current PIL image."""
    global current_pil_image
    current_pil_image = image


def draw_mandelbrot_scanlines_tkinter_palette(canvas, width, height, max_iter, x_min, x_max, y_min, y_max, color_palette, scanlines_update=950):
    """Draw Mandelbrot set with palette coloring."""
    global current_tk_image, current_pil_image, elapsed_time_label
    
    start_time = time.time()  # Start timing
    
    # Calculate all iteration counts at once
    iteration_counts = mandelbrot_set_iterations_pixel_yield_numpy(width, height, max_iter, x_min, x_max, y_min, y_max)
    
    if not fractals_core.drawing_running:  # Check if we should stop
        return
    
    # Update progress for coloring
    if fractals_core.progress_bar and fractals_core.progress_label:
        fractals_core.progress_bar['value'] = 25
        fractals_core.progress_label.config(text="Progress: Coloring...")
        fractals_core.progress_bar.update_idletasks()
    
    if not fractals_core.drawing_running:  # Check if we should stop
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
    if fractals_core.progress_bar and fractals_core.progress_label:
        fractals_core.progress_bar['value'] = 75
        fractals_core.progress_label.config(text="Progress: Rendering...")
        fractals_core.progress_bar.update_idletasks()
    
    # Convert NumPy array to PIL Image
    image = Image.fromarray(img_array)
    current_pil_image = image
    
    # Display the image
    if fractals_core.drawing_running:
        tk_image = pil_image_to_tk(image)
        current_tk_image = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.tk_image = tk_image
        canvas.config(scrollregion=(0, 0, width, height))
        canvas.update()
        
        # Complete progress
        if fractals_core.progress_bar and fractals_core.progress_label:
            fractals_core.progress_bar['value'] = 100
            fractals_core.progress_label.config(text="Progress: Complete")
            fractals_core.progress_bar.update_idletasks()
        
        # Calculate and display elapsed time
        if elapsed_time_label:
            elapsed_time = time.time() - start_time
            elapsed_time_label.config(text=f"{elapsed_time:.2f} sec")


def draw_halley_fractal(canvas, width, height, max_iter, x_min, x_max, y_min, y_max, color_palette, n=3):
    """Draw Halley's fractal on the canvas"""
    global current_tk_image, current_pil_image, elapsed_time_label
    
    start_time = time.time()  # Start timing
    
    # Reset progress bar
    if fractals_core.progress_bar and fractals_core.progress_label:
        fractals_core.progress_bar['value'] = 0
        fractals_core.progress_label.config(text="Progress: Calculating...")
        canvas.update_idletasks()
    
    # Compute iterations and root indices
    iterations, root_indices = halley_fractal_iterations(width, height, max_iter, x_min, x_max, y_min, y_max, n)
    
    # Update progress
    if fractals_core.progress_bar and fractals_core.progress_label:
        fractals_core.progress_bar['value'] = 50
        fractals_core.progress_label.config(text="Progress: Coloring...")
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
    if fractals_core.progress_bar and fractals_core.progress_label:
        fractals_core.progress_bar['value'] = 75
        fractals_core.progress_label.config(text="Progress: Rendering...")
        canvas.update_idletasks()
    
    # Convert the image to PhotoImage and display on canvas
    current_pil_image = img
    
    # Display the image
    if fractals_core.drawing_running:
        tk_image = pil_image_to_tk(img)
        current_tk_image = tk_image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.tk_image = tk_image
        canvas.config(scrollregion=(0, 0, width, height))
        canvas.update()
        
        # Complete progress
        if fractals_core.progress_bar and fractals_core.progress_label:
            fractals_core.progress_bar['value'] = 100
            fractals_core.progress_label.config(text="Progress: Complete")
            canvas.update_idletasks()
        
        # Calculate and display elapsed time
        if elapsed_time_label:
            elapsed_time = time.time() - start_time
            elapsed_time_label.config(text=f"{elapsed_time:.2f} sec")
    
    return iterations, root_indices
