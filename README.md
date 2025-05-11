# KLIC to Shapefiles Converter

A Python utility that converts KLIC/IMKL ZIP deliveries into standardized ESRI Shapefiles. This tool processes Dutch utility location data (KLIC/WIBON) from XML/GML format into readily usable GIS files.

## Features

- Processes single or multiple KLIC ZIP deliveries
- Converts features into standardized shapefile structure
- Preserves EPSG:28992 (Amersfoort/RD New) coordinate system
- Creates all companion files (.shx, .dbf, .prj, .cpg)
- Organizes features into canonical layers:
  - KLIC_Administratief (points, lines, polygons)
  - KLIC_Bijlage (points, lines, polygons)
  - KLIC_EisVoorzorgsmaatregel (polygons)
  - KLIC_LeidingContainerLijn (lines)
  - KLIC_LeidingLijn (lines)
  - KLIC_LeidingPunt (points)

## Requirements

```sh
pip install -r requirements.txt
```

Required packages:
- lxml
- geopandas
- shapely

## Usage

Basic usage with default directories:

```sh
python klic_to_shapefiles.py
```

Specify input and output directories:

```sh
python klic_to_shapefiles.py -i /path/to/input/files -o /path/to/output
```

### Arguments

- `-i`, `--input`: Input ZIP file(s) or directory(ies) containing ZIPs (default: `data/inputs`)
- `-o`, `--output`: Root output directory (default: `data/outputs`)

## Directory Structure

```
project/
├── data/
│   ├── inputs/          # Place KLIC ZIP files here
│   └── outputs/         # Generated shapefiles go here
├── klic_to_shapefiles.py
├── requirements.txt
└── README.md
```

## Output Structure

For each input ZIP file, the script creates a directory with standardized shapefiles:

```
outputs/
└── Levering_XXXXXXXX_1/
    ├── KLIC_Administratief_line.shp
    ├── KLIC_Administratief_point.shp
    ├── KLIC_Administratief_polygon.shp
    ├── KLIC_LeidingLijn.shp
    └── ...
```

Each shapefile comes with its companion files (.shx, .dbf, .prj, .cpg).

## License

[This is a private repository]