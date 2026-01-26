"""
Core fractal calculation functions using NumPy vectorization.
"""
import numpy as np
import threading


# Global state for drawing control
drawing_running = False
progress_bar = None
progress_label = None
show_progress_updates = True  # Toggle for progress bar updates
progress_lock = threading.Lock()  # Lock for thread-safe access


def set_progress_controls(progress_bar_ref, progress_label_ref):
    """Set references to progress bar and label for updates."""
    global progress_bar, progress_label
    progress_bar = progress_bar_ref
    progress_label = progress_label_ref


def set_progress_updates(enabled):
    """Enable or disable progress bar updates (thread-safe)."""
    global show_progress_updates
    with progress_lock:
        show_progress_updates = enabled


def mandelbrot_set_iterations_pixel_yield_numpy(width, height, max_iter, x_min, x_max, y_min, y_max):
    """
    Optimized Mandelbrot set calculation using NumPy vectorization.
    Returns a 2D NumPy array of iteration counts.
    """
    global drawing_running, progress_bar, progress_label, show_progress_updates
    
    # Update progress (always reset to 0, but only show text if toggle is on)
    with progress_lock:
        check_progress = show_progress_updates
    if progress_bar and progress_label:
        progress_bar['value'] = 0
        if check_progress:
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
        
        # Update progress every few iterations (less frequently for better performance)
        with progress_lock:
            check_progress = show_progress_updates
        if check_progress and i % 10 == 0 and progress_bar and progress_label:
            progress = int(20 * (i / max_iter))  # Use first 20% for calculation
            progress_bar['value'] = progress
            progress_label.config(text=f"Progress: Calculating ({i}/{max_iter} iterations)")
            if i % 50 == 0:  # Only update UI every 50 iterations for better performance
                progress_bar.update_idletasks()
        
        if not np.any(mask):
            break
    
    return iterations


def halley_fractal_iterations(width, height, max_iter, x_min, x_max, y_min, y_max, n=3):
    """
    Compute Halley's fractal iterations using NumPy vectorization.
    Returns iteration counts and root indices.
    """
    global drawing_running, progress_bar, progress_label, show_progress_updates
    
    # Update progress (always reset to 0, but only show text if toggle is on)
    with progress_lock:
        check_progress = show_progress_updates
    if progress_bar and progress_label:
        progress_bar['value'] = 0
        if check_progress:
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
        
        # Update progress every few iterations (less frequently for better performance)
        with progress_lock:
            check_progress = show_progress_updates
        if check_progress and i % 10 == 0 and progress_bar and progress_label:
            progress = int(20 * (i / max_iter))  # Use first 20% for calculation
            progress_bar['value'] = progress
            progress_label.config(text=f"Progress: Calculating ({i}/{max_iter} iterations)")
            if i % 50 == 0:  # Only update UI every 50 iterations for better performance
                progress_bar.update_idletasks()
    
    return iterations, root_indices
