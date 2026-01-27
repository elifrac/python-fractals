# Fractal Explorer

A high-performance Tkinter-based application for exploring and visualizing the Mandelbrot set and Halley's fractals in real-time.

## Features

- **Dual Fractal Support**: Explore both the iconic Mandelbrot set and the beautiful Halley's fractals (adjustable power parameter from 2-10)
- **Interactive Zoom**: Click and drag to draw a rectangle around an area to zoom in smoothly
- **Real-time Rendering**: Optimized NumPy vectorization for fast fractal calculations
- **Performance Toggle**: Disable progress updates for significantly faster drawing on large resolutions
- **Customizable Parameters**: 
  - Image resolution (width x height)
  - Iteration depth (up to 255+)
  - Complex plane boundaries (X Min/Max, Y Min/Max)
  - Custom color palettes (MAP format)
- **Parameter Management**: Save and load parameter files as JSON for easy reproduction of interesting regions
- **Image Export**: Save rendered fractals as PNG or JPEG
- **Color Palettes**: Load custom DOS-era palette files for unique color schemes
- **Elapsed Time Tracking**: Monitor rendering performance

## Requirements

- Python 3.6+
- tkinter (usually included with Python)
- NumPy
- Pillow (PIL)

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install numpy pillow
```

3. Run the application:
```bash
python fractals_main.py
```

## Usage

### Starting a Fractal Render

1. Adjust parameters in the right panel:
   - **Width/Height**: Canvas resolution
   - **Max Iterations**: Depth of calculation (higher = more detail but slower)
   - **X Min/Max, Y Min/Max**: Complex plane coordinates to display
2. Select a fractal type via right-click context menu:
   - Mandelbrot
   - Halley's Map (with adjustable power 2-10)
3. Click **"Start Drawing"** to begin rendering
4. The progress bar shows calculation → coloring → rendering stages

### Zooming Into Details

1. Click and drag a rectangle around the area you want to zoom
2. Release to automatically zoom in and redraw
3. The new region's parameters are calculated to maintain aspect ratio

### Managing Palettes

1. Click **"Load Palette"** to select a `.MAP` file from the `Palette/` directory
2. The current palette name appears in the left panel
3. When saving parameters, the palette choice is saved with them

### Saving & Loading

**Save Image**: Export the current fractal as PNG or JPEG
- Saved to: `Fractal Images/` directory

**Save Parameters**: Store render settings for later recreation
- Includes: resolution, iterations, boundaries, palette, fractal type, and power
- Saved to: `Parameter Files/` directory as JSON

**Load Parameters**: Reload previous settings
- Automatically loads the associated palette
- Re-renders the fractal with the loaded parameters

### Performance Optimization

**Uncheck "Show Progress Updates"** to significantly speed up rendering:
- Disables progress bar updates during calculation phase
- Progress bar remains static (no UI overhead)
- Rendering phase still completes normally
- Useful for high-resolution images or high iteration counts

## Controls

| Action | Result |
|--------|--------|
| Click + Drag | Draw zoom rectangle |
| Right Click | Show fractal type menu |
| Start Drawing | Begin render with current parameters |
| End Drawing | Stop current render (partial image shown) |
| Load Defaults | Reset to standard Mandelbrot view |
| Edit Params | Open parameter file in text editor |

## Directory Structure

```
mandelbrot/
├── fractals_main.py          # Main entry point & UI setup
├── fractals_core.py          # Core fractal calculations
├── fractals_render.py        # Image rendering pipeline
├── fractals_ui.py            # UI event handlers & controls
├── fractals_io.py            # File I/O operations
├── fractals_utils.py         # Utility functions
├── Fractal Images/           # Output folder for saved images
├── Parameter Files/          # Saved parameter configurations
├── Palette/                  # Color palette files (.MAP format)
└── README.md                 # This file
```

## File Formats

### Parameter Files (JSON)
```json
{
    "width": 1000,
    "height": 950,
    "max_iter": 255,
    "x_min": "-2.0",
    "x_max": "1.0",
    "y_min": "-1.5",
    "y_max": "1.5",
    "Palette": "GOODEGA2.MAP",
    "fractal_type": "Mandelbrot",
    "halley_power": 3
}
```

### Palette Files (MAP)
Binary or text format containing RGB color data. Can be:
- **Binary**: 3 bytes per color (R, G, B)
- **Text**: Comma-separated RGB values, one per line

## Tips & Tricks

1. **High-Quality Renders**: 
   - Increase iteration count for more detail in boundary regions
   - Higher resolution takes longer but produces better images
   - Uncheck progress updates for faster rendering

2. **Interesting Regions**:
   - The "seahorse valley" around (-0.75, 0.1) in Mandelbrot
   - Zoom into spiral regions for fractal self-similarity

3. **Halley's Fractals**:
   - Try different power values (3-8) for varied patterns
   - Higher powers create more root basins
   - Power 2 creates simple patterns; higher values are more complex

4. **Parameter Files**:
   - Save interesting discoveries for quick access
   - Parameter files include all render settings for reproducibility
   - Share parameter files with others to explore the same regions

5. **Palette Selection**:
   - Different palettes dramatically change the appearance
   - Experiment with various palettes on the same region
   - Create custom palettes in DOS-era MAP format

## Performance

The application uses NumPy vectorization for fast calculations. Performance varies based on:
- **Image resolution**: Larger images take proportionally longer
- **Iteration depth**: Higher iterations = more computation per pixel
- **Complex plane region**: Some areas need more iterations to converge
- **Progress updates**: Enabling them adds UI overhead (~5-15% slower)

Typical render times:
- 1000×950 @ 255 iterations: 1-3 seconds (with progress updates off)
- Same settings with progress updates: 1-4 seconds
- Zoomed regions with high iterations: 5-30+ seconds

## Troubleshooting

**Program won't start**: Ensure all dependencies are installed
```bash
pip install numpy pillow
```

**Palette file not found**: Check that `.MAP` files are in the `Palette/` directory

**Drawing is very slow**: Try unchecking "Show Progress Updates" or reducing resolution/iterations

**Image looks pixelated**: Increase "Max Iterations" for more detail in boundary regions

## Project Structure

This project was created as a learning exercise for Python and GitHub integration. It demonstrates:
- Object-oriented design with modular code organization
- NumPy vectorization for performance
- Tkinter GUI development
- Threading for responsive UI during long calculations
- File I/O operations (JSON, PNG, binary)
- Fractal mathematics and visualization

---

Enjoy exploring the infinite complexity of fractals! 🌀
