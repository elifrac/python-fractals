"""
File I/O operations for saving/loading images, parameters, and palettes.
"""
import os
import json
import shutil
from tkinter import filedialog, messagebox
from PIL import Image
from fractals_utils import pil_image_to_tk
from fractals_render import get_current_pil_image, set_current_pil_image


def save_fractal_image():
    """
    Saves the current fractal image displayed on the canvas.
    Now saves directly from the stored PIL Image, without re-rendering.
    """
    current_pil_image = get_current_pil_image()
    
    # Create Fractal Images directory if it doesn't exist
    images_dir = "Fractal Images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    if current_pil_image:  # Check if there is a PIL image to save
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
                    
               # print(f"Image saved to {file_path}")
            except Exception as e:
                print(f"Error saving image: {e}")
    else:
        print("No image to save. Draw the fractal first.")


def load_fractal_image(canvas):
    """
    Loads a fractal image from a file and displays it on the canvas.
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
            set_current_pil_image(image)  # Store the loaded PIL Image
            canvas.config(width=width, height=height)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor='nw', image=tk_image)
            canvas.tk_image = tk_image
            # Configure the scrollregion to match the image size
            canvas.config(scrollregion=(0, 0, width, height))
           # print(f"Image loaded from {file_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")


def save_parameters_file(width_entry_var, height_entry_var, max_iter_entry_var, 
                           x_min_entry_var, x_max_entry_var, y_min_entry_var, 
                           y_max_entry_var, palette_label=None, current_fractal_type="Mandelbrot", current_halley_power=3):
    """Save fractal parameters to a JSON file."""
    
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
          #  print(f"Parameters saved to {file_path}")
        except Exception as e:
            print(f"Error saving parameters: {e}")


def load_parameters_file(width_entry_var, height_entry_var, max_iter_entry_var, 
                           x_min_entry_var, x_max_entry_var, y_min_entry_var, 
                           y_max_entry_var, palette_label=None, custom_palette=None):
    """Load fractal parameters from a JSON file."""
    import fractals_ui
    
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
                fractals_ui.current_fractal_type = params["fractal_type"]
            if "halley_power" in params:
                fractals_ui.current_halley_power = params["halley_power"]
                
          #  print(f"Parameters loaded from {file_path}")
            
            # Return tuple of (params, file_path) for use by caller
            return (params, file_path)
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return None


def load_palette(palette_file=None, canvas=None, width_entry_var=None, height_entry_var=None, 
                max_iter_entry_var=None, x_min_entry_var=None, x_max_entry_var=None, y_min_entry_var=None, 
                y_max_entry_var=None, palette_label=None, custom_palette=None):
    """Load a color Palette from a file"""
    
    if custom_palette is None:
        return None
    
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
                return None
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
            from fractals_ui import start_drawing_thread
            start_drawing_thread(
                canvas,
                width_entry_var, height_entry_var, max_iter_entry_var,
                x_min_entry_var, x_max_entry_var, y_min_entry_var, y_max_entry_var,
                custom_palette
            )
        
        return custom_palette
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Palette: {str(e)}")
        print(f"Error loading Palette: {e}")
        return None
