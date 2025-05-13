#!/usr/bin/env python3
""" 
This script plots shapefiles with a reference image and saves the output.
It uses the rasterio and geopandas libraries to read and plot the shapefiles and the reference image.
It also uses the lxml library to read the bounding box from an XML file.
"""
import zipfile
import logging
from pathlib import Path
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.transform import Affine
from contextlib import contextmanager
from typing import Generator
from PIL import Image
from lxml import etree
from typing import Tuple, Optional
import argparse
from datetime import datetime
import sys

# Default paths
DEFAULT_INPUT_DIR = Path('data/inputs')
DEFAULT_OUTPUT_DIR = Path('data/extracted')
SHAPEFILES_DIR = Path('data/outputs')
DEFAULT_LOG_DIR = Path('data/logs')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Style definitions for different shapefile types
LAYER_STYLES = {
    'KLIC_Administratief_line': dict(color='dodgerblue'),
    'KLIC_Administratief_polygon': dict(color='blue', alpha=0.3),
    'KLIC_Administratief_point': dict(color='blue', marker='o'),
    'KLIC_Bijlage_line': dict(color='orange'),
    'KLIC_Bijlage_polygon': dict(color='orange', alpha=0.3),
    'KLIC_Bijlage_point': dict(color='orange', marker='s'),
    'KLIC_EisVoorzorgsmaatregel_polygon': dict(color='red', alpha=0.3),
    'KLIC_LeidingContainerLijn_line': dict(color='green'),
    'KLIC_LeidingLijn_line': dict(color='orange'),
    'KLIC_LeidingPunt_point': dict(color='purple', marker='^'),
}

EXPECTED_LAYER_NAMES = list(LAYER_STYLES.keys())

# XML namespaces used in KLIC files
NAMESPACES = {
    'gml': 'http://www.opengis.net/gml/3.2',
    'imkl': 'http://www.geostandaarden.nl/imkl/wibon',
    'net': 'http://inspire.ec.europa.eu/schemas/net/4.0',
    'us-net-common': 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0',
    'xlink': 'http://www.w3.org/1999/xlink'
}

@contextmanager
def iterparse_context(xml_path: str, events=('end',)) -> Generator:
    """Context manager for iterparse that handles cleanup."""
    context = etree.iterparse(xml_path, events=events, recover=True)
    yield context
    del context

def envelope_from_xml(xml_path: str) -> Tuple[float, float, float, float]:
    """Read the bounding box from XML file using streaming parser.
    
    Args:
        xml_path: Path to the XML file
        
    Returns:
        Tuple of (minx, miny, maxx, maxy)
    """
    bbox = {'lower': None, 'upper': None}
    
    try:
        with iterparse_context(xml_path, events=('end',)) as context:
            for event, elem in context:
                if event == 'end':
                    tag = elem.tag.split('}')[-1]  # Remove namespace
                    
                    if tag == 'lowerCorner':
                        bbox['lower'] = list(map(float, elem.text.split()))
                    elif tag == 'upperCorner':
                        bbox['upper'] = list(map(float, elem.text.split()))
                        
                    # Clear element from memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                        
                    # Break if we have both coordinates
                    if bbox['lower'] and bbox['upper']:
                        break
                        
        if not (bbox['lower'] and bbox['upper']):
            raise ValueError("Could not find both lowerCorner and upperCorner in XML")
            
        return bbox['lower'][0], bbox['lower'][1], bbox['upper'][0], bbox['upper'][1]
        
    except Exception as e:
        logger.error(f"Error reading bounding box from {xml_path}: {str(e)}")
        raise

def plot_shapefiles(shapefiles_dir: Path, reference_image_path: Path, 
                   xml_path: Path, save_path: Path) -> None:
    """Plot shapefiles with reference image and save output.
    Creates multiple plots:
    1. Reference image
    2. Polygon layers
    3. Line layers
    4. Point layers
    5. Combined view (all layers)
    
    Args:
        shapefiles_dir: Directory containing shapefiles
        reference_image_path: Path to reference PNG image
        xml_path: Path to XML file containing bounding box
        save_path: Path to save the output plot
    
    Raises:
        Exception: If there's an error during plotting
    """
    try:
        # Read the bounding box from the XML file
        bbox = envelope_from_xml(xml_path)
        logger.info(f"Retrieved bounding box from {xml_path}")

        # Build an affine transform from the bounding box
        minx, miny, maxx, maxy = bbox
        with Image.open(reference_image_path) as img:
            width, height = img.size
        
        px = (maxx - minx) / width
        py = (miny - maxy) / height  # Negative because y-coordinates are inverted in image space
        transform = Affine(px, 0, minx, 0, py, maxy)

        # Create a figure with 5 subplots (2 rows x 3 columns)
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(2, 3, figure=fig)
        axes = [
            fig.add_subplot(gs[0, 0]),  # Reference image
            fig.add_subplot(gs[0, 1]),  # Polygons
            fig.add_subplot(gs[0, 2]),  # Lines
            fig.add_subplot(gs[1, 0]),  # Points
            fig.add_subplot(gs[1, 1:])  # Combined view
        ]
        
        # Function to plot reference image on an axis
        def plot_reference(ax, src_data=None, transform=None):
            if src_data is not None:
                show(src_data, transform=transform, ax=ax)
            ax.set_axis_off()

        # Read reference image once and reuse
        with rasterio.open(reference_image_path, 'r',
                          driver='PNG',
                          transform=transform,
                          crs='EPSG:28992') as src:
            src_data = src.read()
            # Plot reference image on all subplots
            for ax in axes:
                plot_reference(ax, src_data, transform)
            # Clear source data
            del src_data

        # Group layers by type
        polygon_layers = [l for l in EXPECTED_LAYER_NAMES if 'polygon' in l.lower()]
        line_layers = [l for l in EXPECTED_LAYER_NAMES if 'line' in l.lower()]
        point_layers = [l for l in EXPECTED_LAYER_NAMES if 'point' in l.lower()]

        # Function to plot layers of a specific type
        def plot_layers(ax, layer_names, title):
            for layer_name in layer_names:
                shp_path = shapefiles_dir / f"{layer_name}.shp"
                if shp_path.exists():
                    try:
                        # Read shapefile and plot directly
                        style = LAYER_STYLES.get(layer_name, {})
                        gdf = gpd.read_file(shp_path)
                        if not gdf.empty:
                            gdf.plot(ax=ax, **style)
                            logger.info(f"Plotted layer: {layer_name}")
                        
                        # Clear dataframe from memory
                        del gdf
                        
                    except Exception as e:
                        logger.warning(f"Error plotting {layer_name}: {str(e)}")
            ax.set_title(title, pad=10, fontsize=12)

        # Plot each layer type
        plot_layers(axes[0], [], "Reference Image")
        plot_layers(axes[1], polygon_layers, "Polygon Layers")
        plot_layers(axes[2], line_layers, "Line Layers")
        plot_layers(axes[3], point_layers, "Point Layers")
        plot_layers(axes[4], EXPECTED_LAYER_NAMES, "All Layers Combined")

        # Adjust layout and save
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved plot to {save_path}")

    except Exception as e:
        logger.error(f"Error during plotting: {str(e)}")
        raise

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP file to the specified directory.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to
    
    Raises:
        zipfile.BadZipFile: If the file is not a valid ZIP
        OSError: If there are filesystem errors
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except (zipfile.BadZipFile, OSError) as e:
        print(f"Error extracting {zip_path}: {e}")
        raise

def find_corresponding_files(
    base_dir: Path,
    delivery_name: str
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find corresponding XML, PNG and shapefile directory for a delivery.
    
    Args:
        base_dir: Base directory to search in
        delivery_name: Name of the delivery (e.g., 'Levering_25G0132393_1')
    
    Returns:
        Tuple of (xml_path, png_path, shapefile_dir)
    """

    # Look for the XML file with pattern *.xml
    xml_path = None
    if base_dir.exists():
        xml_files = list(base_dir.rglob('*.xml'))
        if xml_files:
            xml_path = xml_files[0]
            logger.info(f"Found XML file: {xml_path}")
    
    # Look for the PNG file with pattern GB_*.png
    png_path = None
    if base_dir.exists():
        png_files = list(base_dir.rglob('GB_*.png'))
        if png_files:
            png_path = png_files[0]
            logger.info(f"Found PNG file: {png_path}")
    
    # Find shapefile directory (should be in output dir with same name)
    shapefile_dir = SHAPEFILES_DIR / delivery_name
    if shapefile_dir.exists():
        logger.info(f"Found shapefile directory: {shapefile_dir}")
    else:
        shapefile_dir = None
        logger.warning(f"Shapefile directory not found: {shapefile_dir}")
    
    return xml_path, png_path, shapefile_dir

def validate_inputs(
    xml_path: Optional[Path],
    png_path: Optional[Path],
    shapefile_dir: Optional[Path]
) -> bool:
    """Validate that all required input files exist.
    
    Args:
        xml_path: Path to XML file
        png_path: Path to PNG file
        shapefile_dir: Directory containing shapefiles
    
    Returns:
        True if all inputs are valid, False otherwise
    """
    if not xml_path or not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        return False
        
    if not png_path or not png_path.exists():
        logger.error(f"PNG file not found: {png_path}")
        return False
        
    if not shapefile_dir or not shapefile_dir.exists():
        logger.error(f"Shapefile directory not found: {shapefile_dir}")
        return False
        
    # Check if at least one shapefile exists
    shp_files = list(shapefile_dir.glob('*.shp'))
    if not shp_files:
        logger.error(f"No shapefiles found in {shapefile_dir}")
        return False
        
    return True

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Plot shapefiles with reference images from KLIC deliveries'
    )
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        default=[str(DEFAULT_INPUT_DIR)],
        help='Input ZIP file(s) or directory(ies) containing ZIPs'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Root output directory'
    )
    parser.add_argument(
        '--plots-dir',
        default='plots',
        help='Directory to save plot images (relative to output directory)'
    )
    args = parser.parse_args()

    # Convert to Path objects
    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output)
    plots_dir = output_dir / args.plots_dir

    # Process each input path
    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix.lower() == '.zip':
            # Single ZIP file
            try:
                delivery_name = input_path.stem
                extract_dir = output_dir / delivery_name
                logger.info(f"Processing delivery: {delivery_name}")
                
                # Extract ZIP if needed
                if not extract_dir.exists():
                    extract_zip(input_path, extract_dir)
                
                # Find required files
                xml_path, png_path, shapefile_dir = find_corresponding_files(
                    extract_dir, delivery_name
                )
                
                # Validate and plot
                if validate_inputs(xml_path, png_path, shapefile_dir):
                    plot_path = plots_dir / f"{delivery_name}_plot.png"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    plot_shapefiles(shapefile_dir, png_path, xml_path, plot_path)
                    logger.info(f"Successfully created plot: {plot_path}")
                
            except Exception as e:
                logger.error(f"Error processing {input_path}: {str(e)}")
                continue
                
        elif input_path.is_dir():
            # Directory containing ZIP files
            for zip_path in input_path.glob('*.zip'):
                try:
                    delivery_name = zip_path.stem
                    extract_dir = output_dir / delivery_name
                    logger.info(f"Processing delivery: {delivery_name}")
                    
                    # Extract ZIP if needed
                    if not extract_dir.exists():
                        extract_zip(zip_path, extract_dir)
                    
                    # Find required files
                    xml_path, png_path, shapefile_dir = find_corresponding_files(
                        extract_dir, delivery_name
                    )
                    
                    # Validate and plot
                    if validate_inputs(xml_path, png_path, shapefile_dir):
                        plot_path = plots_dir / f"{delivery_name}_plot.png"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        plot_shapefiles(shapefile_dir, png_path, xml_path, plot_path)
                        logger.info(f"Successfully created plot: {plot_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {zip_path}: {str(e)}")
                    continue
        else:
            logger.warning(f"Skipping invalid input path: {input_path}")

if __name__ == '__main__':
    try:
        # Setup logging directory
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = DEFAULT_LOG_DIR / f"plot_shapefiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("Starting shapefile plotting process")
        main()
        logger.info("Completed shapefile plotting process")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)