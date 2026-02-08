"""
Utility functions for fractal rendering and image conversion.
"""
import tkinter as tk
import io
try:
    from PIL import Image, ImageTk
    HAVE_IMAGETK = True
except ImportError:
    from PIL import Image
    ImageTk = None
    HAVE_IMAGETK = False


def pil_image_to_tk(image):
    """
    Convert a PIL Image to a Tkinter PhotoImage.
    Uses ImageTk when available, otherwise falls back to tk.PhotoImage via in-memory PNG.
    """
    if HAVE_IMAGETK and ImageTk is not None:
        return ImageTk.PhotoImage(image=image)
    else:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            data = buffer.getvalue()
        return tk.PhotoImage(data=data)
