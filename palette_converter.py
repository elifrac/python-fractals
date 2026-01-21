import os
import sys
import glob

def unpack_rgb(packed_value, format="auto"):
    """
    Unpack an integer RGB value into its R, G, B components.
    
    Args:
        packed_value (int): The packed RGB value
        format (str): The format of the packed value:
            - "auto": Auto-detect format (default)
            - "rgb888": 8 bits per channel (0xRRGGBB)
            - "bgr888": 8 bits per channel, reversed (0xBBGGRR)
            - "rgb565": 5 bits R, 6 bits G, 5 bits B
            - "rgb555": 5 bits per channel
    
    Returns:
        tuple: (R, G, B) values
    """
    try:
        # Try to convert the value to an integer
        value = int(packed_value.strip())
        
        # Auto-detect format based on value range
        if format == "auto":
            if value <= 0xFFFFFF:  # 24-bit color
                format = "rgb888"
            elif value <= 0x7FFF:  # 15-bit color
                format = "rgb555"
            else:
                format = "rgb888"  # Default to standard RGB
        
        # Extract RGB components based on format
        if format == "rgb888":
            r = (value >> 16) & 0xFF
            g = (value >> 8) & 0xFF
            b = value & 0xFF
        elif format == "bgr888":
            b = (value >> 16) & 0xFF
            g = (value >> 8) & 0xFF
            r = value & 0xFF
        elif format == "rgb565":
            r = ((value >> 11) & 0x1F) * 255 // 31
            g = ((value >> 5) & 0x3F) * 255 // 63
            b = (value & 0x1F) * 255 // 31
        elif format == "rgb555":
            r = ((value >> 10) & 0x1F) * 255 // 31
            g = ((value >> 5) & 0x1F) * 255 // 31
            b = (value & 0x1F) * 255 // 31
        else:
            # Default to standard RGB
            r = (value >> 16) & 0xFF
            g = (value >> 8) & 0xFF
            b = value & 0xFF
        
        return (r, g, b)
    except ValueError:
        # If conversion fails, return None
        return None

def convert_file(input_file, output_file=None):
    """
    Convert space-separated RGB values to comma-separated values.
    
    Args:
        input_file (str): Path to the input file
        output_file (str, optional): Path to the output file. If None, will use input_file with '_csv' suffix
    
    Returns:
        str: Path to the output file
    """
    if output_file is None:
        # Create output filename by adding '_csv' before the extension
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_csv{ext}"
    
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
    
    # Determine if this is a packed integer file or space-separated RGB file
    # Check the first few non-empty lines to see if they contain single values or multiple values
    is_packed_format = True
    for line in lines[:10]:  # Check first 10 lines
        if line.strip():  # Skip empty lines
            values = [val for val in line.strip().split() if val]
            if len(values) > 1:
                is_packed_format = False
                break
    
    converted_lines = []
    if is_packed_format:
        # Handle packed integer format
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            rgb = unpack_rgb(line)
            if rgb:
                converted_line = f"{rgb[0]},{rgb[1]},{rgb[2]}"
                converted_lines.append(converted_line)
            else:
                # Keep original line if unpacking fails
                converted_lines.append(line)
    else:
        # Handle space-separated RGB format
        for line in lines:
            # Strip whitespace and split by any number of spaces
            values = [val for val in line.strip().split() if val]
            if len(values) == 3:  # Only process lines with exactly 3 values (RGB)
                converted_line = ','.join(values)
                converted_lines.append(converted_line)
            else:
                # Keep original line if it doesn't have exactly 3 values
                converted_lines.append(line.strip())
    
    with open(output_file, 'w') as f_out:
        f_out.write('\n'.join(converted_lines))
    
    return output_file

def convert_pal_to_map(input_file, output_file=None, rgb_format="auto"):
    """
    Convert a PAL file with packed integer RGB values to a MAP file with comma-separated RGB values.
    
    Args:
        input_file (str): Path to the input PAL file
        output_file (str, optional): Path to the output MAP file. If None, will create one with the same base name
        rgb_format (str): The format of the packed RGB values (see unpack_rgb function)
    
    Returns:
        str: Path to the output MAP file
    """
    if output_file is None:
        # Create output filename by changing extension to .MAP
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}.MAP"
    
    with open(input_file, 'r') as f_in:
        lines = f_in.readlines()
    
    converted_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        rgb = unpack_rgb(line, rgb_format)
        if rgb:
            converted_line = f"{rgb[0]},{rgb[1]},{rgb[2]}"
            converted_lines.append(converted_line)
    
    with open(output_file, 'w') as f_out:
        f_out.write('\n'.join(converted_lines))
    
    return output_file

def convert_directory(directory, pattern="*.MAP"):
    """
    Convert all files matching the pattern in the directory.
    
    Args:
        directory (str): Directory containing palette files
        pattern (str): Glob pattern to match files
    
    Returns:
        list: List of converted file paths
    """
    converted_files = []
    for file_path in glob.glob(os.path.join(directory, pattern)):
        output_path = convert_file(file_path)
        converted_files.append(output_path)
        print(f"Converted {file_path} to {output_path}")
    
    return converted_files

def convert_pal_directory(directory, pattern="*.PAL", rgb_format="auto"):
    """
    Convert all PAL files matching the pattern in the directory to MAP files.
    
    Args:
        directory (str): Directory containing PAL files
        pattern (str): Glob pattern to match files
        rgb_format (str): The format of the packed RGB values (see unpack_rgb function)
    
    Returns:
        list: List of converted file paths
    """
    converted_files = []
    for file_path in glob.glob(os.path.join(directory, pattern)):
        output_path = convert_pal_to_map(file_path, None, rgb_format)
        converted_files.append(output_path)
        print(f"Converted PAL file {file_path} to MAP file {output_path}")
    
    return converted_files

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python palette_converter.py <input_file> [output_file]")
        print("  Directory:   python palette_converter.py --dir <directory> [--pattern '*.MAP']")
        print("  PAL to MAP:  python palette_converter.py --pal-to-map <input_file> [output_file] [--format FORMAT]")
        print("  PAL Dir:     python palette_converter.py --pal-dir <directory> [--pattern '*.PAL'] [--format FORMAT]")
        print("\nAvailable formats for PAL conversion:")
        print("  auto   - Auto-detect format (default)")
        print("  rgb888 - 8 bits per channel (0xRRGGBB)")
        print("  bgr888 - 8 bits per channel, reversed (0xBBGGRR)")
        print("  rgb565 - 5 bits R, 6 bits G, 5 bits B")
        print("  rgb555 - 5 bits per channel")
        return
    
    if sys.argv[1] == "--dir":
        if len(sys.argv) < 3:
            print("Error: Directory path required with --dir option")
            return
        
        directory = sys.argv[2]
        pattern = "*.MAP"  # Default pattern
        
        if len(sys.argv) > 3 and sys.argv[3] == "--pattern" and len(sys.argv) > 4:
            pattern = sys.argv[4]
        
        converted = convert_directory(directory, pattern)
        print(f"Converted {len(converted)} files in {directory}")
    elif sys.argv[1] == "--pal-to-map":
        if len(sys.argv) < 3:
            print("Error: Input file path required with --pal-to-map option")
            return
        
        input_file = sys.argv[2]
        output_file = None
        rgb_format = "auto"
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--format" and i + 1 < len(sys.argv):
                rgb_format = sys.argv[i + 1]
                i += 2
            else:
                output_file = sys.argv[i]
                i += 1
        
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            return
        
        output_path = convert_pal_to_map(input_file, output_file, rgb_format)
        print(f"Converted PAL file {input_file} to MAP file {output_path}")
    elif sys.argv[1] == "--pal-dir":
        if len(sys.argv) < 3:
            print("Error: Directory path required with --pal-dir option")
            return
        
        directory = sys.argv[2]
        pattern = "*.PAL"
        rgb_format = "auto"
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--pattern" and i + 1 < len(sys.argv):
                pattern = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--format" and i + 1 < len(sys.argv):
                rgb_format = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        # Update convert_pal_directory to use the format
        converted = []
        for file_path in glob.glob(os.path.join(directory, pattern)):
            output_path = convert_pal_to_map(file_path, None, rgb_format)
            converted.append(output_path)
            print(f"Converted PAL file {file_path} to MAP file {output_path}")
        
        print(f"Converted {len(converted)} PAL files to MAP files in {directory}")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found")
            return
        
        output_path = convert_file(input_file, output_file)
        print(f"Converted {input_file} to {output_path}")

if __name__ == "__main__":
    main() 