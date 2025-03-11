import os
import json
import ifcopenshell
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_ifc_data(ifc_file_path):
    """
    Extracts data from an IFC file and returns a structured dictionary.

    Args:
        ifc_file_path (str): Path to the IFC file.

    Returns:
        dict: A dictionary containing extracted data such as project name, materials, spatial info,
              element counts, and quantities.

    Raises:
        Exception: If the IFC file cannot be processed.
    """
    try:
        start_time = datetime.now()
        file_size = os.path.getsize(ifc_file_path) / (1024 * 1024)  # Size in MB
        logging.info(f"Processing file: {ifc_file_path} (Size: {file_size:.2f} MB)")

        ifc_file = ifcopenshell.open(ifc_file_path)

        # Extract project name
        project_name = ifc_file.by_type("IfcProject")[0].Name if ifc_file.by_type("IfcProject") else "Unknown Project"

        # Count entities
        entity_counts = {}
        for entity in ifc_file:
            entity_name = entity.is_a()
            entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1

        # Extract materials
        materials = {mat.Name for mat in ifc_file.by_type("IfcMaterial")}

        # Extract spatial structure
        sites = [site.Name for site in ifc_file.by_type("IfcSite")]
        buildings = [bld.Name for bld in ifc_file.by_type("IfcBuilding")]
        stories = [story.Name for story in ifc_file.by_type("IfcBuildingStorey")]

        # Extract element counts
        element_counts = {
            "walls": len(ifc_file.by_type("IfcWall")),
            "slabs": len(ifc_file.by_type("IfcSlab")),
            "roofs": len(ifc_file.by_type("IfcRoof")),
            "columns": len(ifc_file.by_type("IfcColumn")),
            "beams": len(ifc_file.by_type("IfcBeam")),
            "windows": len(ifc_file.by_type("IfcWindow")),
            "doors": len(ifc_file.by_type("IfcDoor")),
        }

        # Helper function to extract quantities for an element type
        def get_element_quantities(element_type, quantity_type):
            """
            Calculates the total quantity (area or volume) for a given element type.

            Args:
                element_type (str): The type of element (e.g., "IfcWall").
                quantity_type (str): The type of quantity (e.g., "IfcQuantityArea").

            Returns:
                float: The total quantity value.
            """
            total_value = 0
            for element in ifc_file.by_type(element_type):
                rels = ifc_file.get_inverse(element)
                for rel in rels:
                    if rel.is_a("IfcRelDefinesByProperties"):
                        prop_set = rel.RelatingPropertyDefinition
                        if prop_set.is_a("IfcElementQuantity"):
                            for quantity in prop_set.Quantities:
                                if quantity.is_a(quantity_type):
                                    total_value += getattr(quantity, "AreaValue", 0) if quantity_type == "IfcQuantityArea" else getattr(quantity, "VolumeValue", 0)
            return total_value

        # Calculate quantities
        total_wall_area = get_element_quantities("IfcWall", "IfcQuantityArea")
        total_slab_area = get_element_quantities("IfcSlab", "IfcQuantityArea")
        total_roof_area = get_element_quantities("IfcRoof", "IfcQuantityArea")
        total_plastering_area = total_wall_area
        total_concrete_volume = get_element_quantities("IfcSlab", "IfcQuantityVolume") + get_element_quantities("IfcWall", "IfcQuantityVolume")

        processing_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Finished processing {ifc_file_path} in {processing_time:.2f} seconds")

        return {
            "file_name": os.path.basename(ifc_file_path),
            "project_name": project_name,
            "materials": list(materials),
            "spatial_info": {
                "sites": sites,
                "buildings": buildings,
                "stories": stories,
            },
            "element_counts": element_counts,
            "quantities": {
                "total_wall_area_m2": total_wall_area,
                "total_slab_area_m2": total_slab_area,
                "total_roof_area_m2": total_roof_area,
                "total_plastering_area_m2": total_plastering_area,
                "total_concrete_volume_m3": total_concrete_volume,
            }
        }
    except Exception as e:
        logging.error(f"Error extracting data from {ifc_file_path}: {e}")
        raise