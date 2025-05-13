#!/usr/bin/env python3
"""
XML Analysis tool using streaming parser for large XML files.
Uses iterparse to process XML elements one at a time without loading entire file into memory.
"""
from lxml import etree
from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict, Optional, Generator, Tuple
from contextlib import contextmanager
import argparse
import zipfile

# Default paths
DEFAULT_INPUT_DIR = Path('data/inputs')
DEFAULT_OUTPUT_DIR = Path('data/reports')
DEFAULT_LOG_DIR = Path('data/logs')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# XML namespaces used in KLIC files
NAMESPACES: Dict[str, str] = {
    'gml': 'http://www.opengis.net/gml/3.2',
    'imkl': 'http://www.geostandaarden.nl/imkl/wibon',
    'net': 'http://inspire.ec.europa.eu/schemas/net/4.0',
    'us-net-common': 'http://inspire.ec.europa.eu/schemas/us-net-common/4.0',
    'xlink': 'http://www.w3.org/1999/xlink'
}

# Feature mappings from refined requirements
KLIC_FEATURE_MAPPINGS = {
    'KLIC_Administratief': {
        'elements': ['Graafpolygoon', 'Annotatie', 'Maatvoering', 'TechnischGebouw'],
        'schema': 'Leveringsinformatie-2.1.xsd',
        'geom_types': ['point', 'line', 'polygon']
    },
    'KLIC_Bijlage': {
        'elements': ['Bijlage'],
        'schema': 'KlicDocumentenBeheer-1.0.xsd',
        'geom_types': ['point', 'line', 'polygon']
    },
    'KLIC_EisVoorzorgsmaatregel': {
        'elements': ['EisVoorzorgsmaatregel', 'Voorzorgsmaatregel'],
        'schema': 'KlicVoorzorgsmaatregelenBeheer-1.0.xsd',
        'geom_types': ['polygon']
    },
    'KLIC_LeidingContainerLijn': {
        'elements': ['LeidingContainerLijn'],
        'schema': 'Leveringsinformatie-2.1.xsd',
        'geom_types': ['line']
    },
    'KLIC_LeidingLijn': {
        'elements': ['LeidingLijn'],
        'schema': 'Leveringsinformatie-2.1.xsd',
        'geom_types': ['line']
    },
    'KLIC_LeidingPunt': {
        'elements': ['LeidingPunt'],
        'schema': 'Leveringsinformatie-2.1.xsd',
        'geom_types': ['point']
    }
}

class StreamingKLICAnalyzer:
    """Memory-efficient KLIC XML analyzer using iterative parsing."""

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.feature_counts = defaultdict(int)
        self.geometry_types = defaultdict(set)
        # count per‑group geometries!
        self.klic_group_features = defaultdict(lambda: defaultdict(int))
        self.crs_info = None

        # precompute lowercase element names for grouping
        self.group_elements = {
            group: [e.lower() for e in info['elements']]
            for group, info in KLIC_FEATURE_MAPPINGS.items()
        }
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Get target elements for parsing
        self.target_elements = self._get_target_elements()

    def _get_target_elements(self) -> set:
        """Get all elements we need to track from KLIC mappings."""

        # lower-case the tags we care about
        elements = set()
        for group_info in KLIC_FEATURE_MAPPINGS.values():
            for e in group_info['elements']:
                elements.add(e.lower())
        # lower-case the extra “always-track” tags
        for e in [
            'ExtraGeometrie','UtilityLink','Appurtenance',
            'Belanghebbende','Belang','Beheerder','GebiedsinformatieLevering'
        ]:
            elements.add(e.lower())
        return elements


    @contextmanager
    def _iterparse(self, xml_path: str, events=('end',)) -> Generator:
        """Context manager for iterparse that handles cleanup."""
        context = etree.iterparse(xml_path, events=events, recover=True)
        yield context
        del context

    def _tag(self, elem):
        # helper to get lowercase local-name
        return elem.tag.split('}')[-1].lower()




    def _get_geometry_type(self, element) -> Optional[str]:
        """Determine geometry type from any GML or envelope-like tag underneath."""
        geom_checks = {
            'point':     ['Point', 'MultiPoint', 'pos', 'posList'],
            'line':      ['LineString', 'MultiLineString', 'Curve', 'MultiCurve',
                          'LineStringSegment', 'LinearRing'],
            'polygon':   ['Polygon', 'MultiPolygon', 'Surface', 'MultiSurface',
                          'Envelope', 'boundedBy', 'exterior', 'interior']
        }

        # scan all descendants
        for desc in element.iter():
            local = desc.tag.split('}')[-1]
            for geom_type, tagset in geom_checks.items():
                if local in tagset:
                    return geom_type
        return None

    def analyze(self):
        self.logger.info(f"Starting streaming analysis of: {self.xml_path}")
        ns = {**NAMESPACES, 'gml': NAMESPACES['gml']}

        # we only care when a featureMember closes
        for _, fm in etree.iterparse(self.xml_path, events=('end',),
                                     tag=f"{{{ns['gml']}}}featureMember",
                                     recover=True):
            # fm is <gml:featureMember>…</gml:featureMember>
            # its first (and only) child is the real feature
            feature = fm[0]
            feat_tag = feature.tag.split('}')[-1]  # e.g. “Annotatie” or “Maatvoering”
            feat_key = feat_tag.lower()

            # count if it's one we track
            if feat_key in self.target_elements:
                self.feature_counts[feat_key] += 1

                # look for geometry under two possible places
                # 1) point/linestring under imkl:ligging
                geom = feature.find('.//gml:Point',   namespaces=ns) \
                    or feature.find('.//gml:LineString', namespaces=ns) \
                    or feature.find('.//gml:Polygon',    namespaces=ns)

                # 2) if it’s a “vlakgeometrie2D” feature (e.g. ExtraGeometrie),
                #    the real Polygon lives deeper
                if geom is None:
                    geom = feature.find('.//imkl:vlakgeometrie2D//gml:Polygon', namespaces=ns)

                if geom is not None:
                    tag = geom.tag.split('}')[-1]
                    # map the real tag to our abstract type
                    if tag == 'Point':
                        geom_type = 'point'
                    elif tag == 'LineString':
                        geom_type = 'line'
                    elif tag == 'Polygon':
                        geom_type = 'polygon'
                    else:
                        geom_type = None

                    if geom_type:
                        self.geometry_types[feat_key].add(geom_type)
                        # increment the right group bucket
                        for group_name, elems in self.group_elements.items():
                            if feat_key in elems:
                                self.klic_group_features[group_name][geom_type] += 1

            # clear the whole featureMember from memory
            fm.clear()
            while fm.getprevious() is not None:
                del fm.getparent()[0]

        return True

    def generate_report(self) -> str:
        """Generate analysis report."""
        report = []
        report.append("KLIC XML Streaming Analysis Report")
        report.append("================================\n")

        # 1. Feature Counts
        report.append("1. Feature Counts:")
        report.append("-----------------")
        for feature, count in sorted(self.feature_counts.items()):
            if count > 0:
                report.append(f"{feature}: {count}")
        report.append("")

        # 2. KLIC Group Analysis
        report.append("2. KLIC Group Analysis:")
        report.append("--------------------")
        for group_name, group_info in KLIC_FEATURE_MAPPINGS.items():
            report.append(f"\n{group_name}:")
            report.append(f"  Schema: {group_info['schema']}")
            report.append("  Elements:")
            for element in group_info['elements']:
                key = element.lower()
                count = self.feature_counts.get(key, 0)
                status = "✓" if count > 0 else "-"
                report.append(f"    {element}: {status} ({count})")
            report.append("  Geometry Types:")
            for geom_type in group_info['geom_types']:
                count = self.klic_group_features[group_name].get(geom_type, 0)
                status = "✓" if count > 0 else "-"
                report.append(f"    {geom_type}: {status} ({count})")
        report.append("")

        # 3. Geometry Types by Feature
        report.append("3. Geometry Types by Feature:")
        report.append("--------------------------")
        for feature, geom_types in sorted(self.geometry_types.items()):
            if geom_types:
                report.append(f"{feature}: {', '.join(sorted(geom_types))}")
        report.append("")

        # 4. CRS Information
        report.append("4. CRS Information:")
        report.append("-----------------")
        report.append(f"CRS: {self.crs_info or 'Not specified'}")

        return "\n".join(report)


def find_corresponding_files(
    base_dir: Path,
    delivery_name: str
) -> Optional[Path]:
    """Find XML file for a delivery.
    
    Args:
        base_dir: Base directory to search in
        delivery_name: Name of the delivery (e.g., 'Levering_25G0132393_1')
    
    Returns:
        Optional[Path] to the XML file
    """
    # Look for XML files with pattern GI_gebiedsinformatielevering_*.xml
    xml_path = None
    if base_dir.exists():
        # Search in the expected structure: base_dir/delivery_name/delivery_id/...
        delivery_number = delivery_name.replace('Levering_', '')
        for xml_file in base_dir.rglob('*.xml'):
            if ('GI_gebiedsinformatielevering' in xml_file.name and 
                delivery_number in xml_file.name):
                xml_path = xml_file
                logger.info(f"Found XML file: {xml_path}")
                break
    
    return xml_path

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

def validate_inputs(
    xml_path: Optional[Path]
) -> bool:
    """Validate that all required input files exist.
    
    Args:
        xml_path: Path to XML file
    
    Returns:
        True if all inputs are valid, False otherwise
    """
    if not xml_path or not xml_path.exists():
        logger.error(f"XML file not found: {xml_path}")
        return False
        
    return True

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='KLIC XML Analysis Tool',
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

    # Convert to Path objects
    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each input path
    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix.lower() == '.zip':
            # Process single ZIP file
            process_delivery(input_path, output_dir)
        elif input_path.is_dir():
            # Process directory of ZIP files
            for zip_path in input_path.glob('*.zip'):
                process_delivery(zip_path, output_dir)

def process_delivery(zip_path: Path, output_dir: Path) -> None:
    """Process a single delivery ZIP file."""
    try:
        delivery_name = zip_path.stem
        extract_dir = Path('data/extracted') / delivery_name
        logger.info(f"Processing delivery: {delivery_name}")
        
        # Extract ZIP if needed
        if not extract_dir.exists():
            extract_zip(zip_path, extract_dir)
        
        # Find XML file
        xml_path = find_corresponding_files(extract_dir, delivery_name)
        
        # Validate and generate reports
        if validate_inputs(xml_path):
            analyzer = StreamingKLICAnalyzer(str(xml_path))
            if analyzer.analyze():
                report = analyzer.generate_report()
                report_path = output_dir / f"{delivery_name}_analysis.txt"
                with open(report_path, 'w') as report_file:
                    report_file.write(report)
                logger.info(f"Report saved to: {report_path}")
            else:
                logger.error(f"Failed to analyze XML file: {xml_path}")
        else:
            logger.error(f"Input validation failed for delivery: {delivery_name}")
    except Exception as e:
        logger.error(f"Error processing {zip_path}: {e}")
        raise

if __name__ == "__main__":
    main()
