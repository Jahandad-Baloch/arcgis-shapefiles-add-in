#!/usr/bin/env python3
"""KLIC XML to Shapefile Converter

This script processes KLIC/IMKL XML/GML data from ZIP deliveries and exports
features into standardized ESRI Shapefiles. Features are organized into canonical
groups with proper geometry types (points, lines, polygons).

Key Features:
- Processes single or multiple KLIC ZIP deliveries
- Exports to standardized shapefile structure
- Preserves EPSG:28992 coordinate system
- Creates all companion files (.shx, .dbf, .prj, .cpg)
"""
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Optional, Tuple, Dict, Any, List
from lxml import etree
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import zipfile


# XML namespaces used in KLIC files
NAMESPACES: Dict[str, str] = {
    'gml': 'http://www.opengis.net/gml/3.2',
    'imkl': 'http://www.geostandaarden.nl/imkl/wibon',
    'net': 'http://inspire.ec.europa.eu/schemas/net/4.0',
    'us-net-common': 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0',
    'xlink': 'http://www.w3.org/1999/xlink'
}

# Default paths
DEFAULT_INPUT_DIR = Path('data/inputs')
DEFAULT_OUTPUT_DIR = Path('data/outputs')
DEFAULT_LOG_DIR = Path('data/logs')

# Canonical layer definitions
CANONICAL_LAYERS = {
    'KLIC_Administratief': ['point', 'line', 'polygon'],
    'KLIC_Bijlage': ['point', 'line', 'polygon'],
    'KLIC_EisVoorzorgsmaatregel': ['polygon'],
    'KLIC_LeidingContainerLijn': ['line'],
    'KLIC_LeidingLijn': ['line'],
    'KLIC_LeidingPunt': ['point']
}

# Tags to skip during processing
SKIP_TAGS = {'GebiedsinformatieLevering', 'GebiedsinformatieAanvraag', 'Belanghebbende'}

# Default coordinate reference system
DEFAULT_CRS = 'EPSG:28992'  # Amersfoort/RD New


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
        raise

def clean_tag(tag: str) -> str:
    """Remove namespace from XML tag.
    
    Args:
        tag: The XML tag possibly containing a namespace
    
    Returns:
        The tag with namespace removed
    """
    return tag.split('}')[-1] if '}' in tag else tag

def find_xml_files(directory: Path) -> List[Path]:
    """Find all XML files in directory recursively.
    
    Args:
        directory: Directory to search in
    
    Returns:
        List of paths to XML files
    """
    return list(directory.rglob('*.xml'))

class ShapefileConverter:
    """Converts KLIC/IMKL features to shapefiles."""
    
    def __init__(self, output_dir: Path):
        """Initialize converter with output directory.
        
        Args:
            output_dir: Directory where shapefiles will be created
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_collections: Dict[str, List[dict]] = {
            f"{group}_{geom}": []
            for group, geoms in CANONICAL_LAYERS.items()
            for geom in geoms
        }
    
    def add_feature(self, feature_type: str, geometry: Point | LineString | Polygon,
                   attributes: dict, geom_type: str) -> None:
        """Add a feature to be written to shapefile.
        
        Args:
            feature_type: Canonical feature type (e.g., 'KLIC_Administratief')
            geometry: Shapely geometry object
            attributes: Feature attributes/properties
            geom_type: Geometry type ('point', 'line', or 'polygon')
        """
        if geom_type not in ['point', 'line', 'polygon']:
            return
        
        collection_key = f"{feature_type}_{geom_type}"
        if collection_key not in self.feature_collections:
            return
        
        self.feature_collections[collection_key].append({
            'geometry': geometry,
            'properties': attributes
        })
    
    def write_shapefiles(self) -> None:
        """Write all collected features to shapefiles."""
        for collection_name, features in self.feature_collections.items():
            if not features:
                # Create empty shapefile to maintain consistent structure
                self._write_empty_shapefile(collection_name)
                continue
            
            try:
                gdf = gpd.GeoDataFrame.from_features(features, crs=DEFAULT_CRS)
                output_path = self.output_dir / f"{collection_name}.shp"
                gdf.to_file(output_path, encoding='utf-8')
            except Exception as e:
                continue

    def _write_empty_shapefile(self, name: str) -> None:
        """Create an empty shapefile with correct schema.
        
        Args:
            name: Name of the shapefile (without extension)
        """
        geom_type = name.split('_')[-1]
        geometry_type = {
            'point': 'Point',
            'line': 'LineString',
            'polygon': 'Polygon'
        }[geom_type]
        
        gdf = gpd.GeoDataFrame(
            columns=['geometry'],
            geometry=[],
            crs=DEFAULT_CRS
        )
        gdf.geometry = gdf.geometry.astype(geometry_type)
        
        output_path = self.output_dir / f"{name}.shp"
        try:
            gdf.to_file(output_path, encoding='utf-8')
        except Exception as e:
            pass

def find_imkl_xmls(root: Path) -> List[Path]:
    """Find all IMKL delivery XMLs under root.
    
    Args:
        root: Root directory to search in
    
    Returns:
        List of paths to XML files
    """
    return [p for p in root.rglob('*.xml') 
            if 'gebiedsinformatielevering' in p.name.lower()]

def parse_imkl_xml(xml_path: Path) -> Generator[Tuple[str, str, Any, Dict], None, None]:
    """Parse XML and yield feature information.
    
    Args:
        xml_path: Path to XML file
    
    Yields:
        Tuples of (tag, geometry_type, geometry_object, attributes)
    """
    tree = etree.parse(str(xml_path))
    
    for feature_member in tree.findall('.//gml:featureMember', namespaces=NAMESPACES):
        for element in feature_member:
            tag = clean_tag(element.tag)
            if tag in SKIP_TAGS:
                continue
            
            # Extract geometry
            geom_obj = None
            geom_type = None
            
            # Try Point geometry
            point = element.find('.//gml:Point', namespaces=NAMESPACES)
            if point is not None:
                pos = point.find('.//gml:pos', namespaces=NAMESPACES)
                if pos is not None:
                    x, y = map(float, pos.text.split())
                    geom_obj, geom_type = Point(x, y), 'point'
            
            # Try LineString geometry
            if not geom_obj:
                line = element.find('.//gml:LineString', namespaces=NAMESPACES)
                if line is not None:
                    pos_list = line.find('.//gml:posList', namespaces=NAMESPACES)
                    if pos_list is not None:
                        coords = list(map(float, pos_list.text.split()))
                        points = [(coords[i], coords[i+1]) 
                                for i in range(0, len(coords), 2)]
                        geom_obj, geom_type = LineString(points), 'line'
            
            # Try Polygon geometry
            if not geom_obj:
                polygon = element.find('.//gml:Polygon', namespaces=NAMESPACES)
                if polygon is not None:
                    exterior = polygon.find('.//gml:exterior//gml:posList',
                                        namespaces=NAMESPACES)
                    if exterior is not None:
                        coords = list(map(float, exterior.text.split()))
                        shell = [(coords[i], coords[i+1]) 
                                for i in range(0, len(coords), 2)]
                        
                        # Handle interior rings (holes)
                        holes = []
                        for interior in polygon.findall('.//gml:interior//gml:posList',
                                                      namespaces=NAMESPACES):
                            coords = list(map(float, interior.text.split()))
                            hole = [(coords[i], coords[i+1]) 
                                  for i in range(0, len(coords), 2)]
                            holes.append(hole)
                        
                        geom_obj, geom_type = Polygon(shell, holes), 'polygon'
            
            if not geom_obj:
                continue
            
            # Extract attributes
            attrs = {}
            if gml_id := element.get(f'{{{NAMESPACES["gml"]}}}id'):
                attrs['gml_id'] = gml_id
            
            for child in element:
                child_name = clean_tag(child.tag)
                if child_name in {'Point', 'LineString', 'Polygon', 'exterior', 'interior'}:
                    continue
                
                if href := child.get(f'{{{NAMESPACES["xlink"]}}}href'):
                    attrs[child_name] = href
                elif child.text and (text := child.text.strip()):
                    attrs[child_name] = text
            
            yield tag, geom_type, geom_obj, attrs

def map_to_canonical_layer(tag: str, geom_type: str) -> Optional[str]:
    """Map source tag and geometry type to canonical layer.
    
    Args:
        tag: Feature tag name
        geom_type: Geometry type ('point', 'line', or 'polygon')
    
    Returns:
        Canonical layer name or None if no mapping found
    """
    tag = tag.lower()
    
    # Administrative features
    if any(x in tag for x in ['graafpolygoon', 'annotatie', 'maatvoering', 'technischgebouw']):
        return 'KLIC_Administratief'
    
    # Bijlage features
    if 'bijlage' in tag:
        return 'KLIC_Bijlage'
    
    # EisVoorzorgsmaatregel features
    if 'eisvoorzorgsmaatregel' in tag:
        return 'KLIC_EisVoorzorgsmaatregel'
    
    # Utility features
    if 'utilitylink' in tag or 'leidinglijn' in tag:
        return 'KLIC_LeidingLijn'
    if 'extrageometrie' in tag or 'leidingcontainer' in tag:
        return 'KLIC_LeidingContainerLijn'
    if 'appurtenance' in tag or 'leidingpunt' in tag:
        return 'KLIC_LeidingPunt'
    
    # Fallback matches
    if 'leiding' in tag and geom_type == 'line':
        return 'KLIC_LeidingLijn'
    if 'punt' in tag or geom_type == 'point':
        return 'KLIC_LeidingPunt'
    
    return None

def process_delivery(xml_path: Path, converter: ShapefileConverter) -> None:
    """Process a single KLIC delivery XML file.
    
    Args:
        xml_path: Path to XML file
        converter: ShapefileConverter instance
    """
    try:
        for tag, geom_type, geom, attrs in parse_imkl_xml(xml_path):
            if canonical := map_to_canonical_layer(tag, geom_type):
                # Truncate attribute names to 10 chars for shapefile compatibility
                attrs = {k[:10]: v for k, v in attrs.items()}
                converter.add_feature(canonical, geom, attrs, geom_type)
    except Exception as e:
        return

def process_zip(zip_path: Path, out_root: Path) -> None:
    """Process a single KLIC ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        out_root: Root output directory
    """
    tmp = Path(tempfile.mkdtemp(prefix='klic_'))
    
    try:
        # Extract ZIP
        extract_zip(zip_path, tmp)
        xml_files = find_imkl_xmls(tmp)
        
        if not xml_files:
            return
        
        # Set up output directory and converter
        out_dir = out_root / zip_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        converter = ShapefileConverter(out_dir)
        
        # Process each XML file
        for xml_path in xml_files:
            process_delivery(xml_path, converter)
        
        # Write shapefiles
        converter.write_shapefiles()
    
    except Exception as e:
        return
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert KLIC ZIP files to Shapefiles'
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
    args = parser.parse_args()
    
    # Process input paths
    input_paths = []
    for path_str in args.input:
        path = Path(path_str)
        if path.is_dir():
            input_paths.extend(path.glob('*.zip'))
        elif path.is_file() and path.suffix.lower() == '.zip':
            input_paths.append(path)
        else:
            return
    
    if not input_paths:
        return
    
    # Create output directory
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Process each ZIP file
    for zip_path in sorted(input_paths):
        process_zip(zip_path, out_root)

if __name__ == "__main__":
    main()
