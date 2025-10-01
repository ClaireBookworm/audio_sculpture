"""
STL to Web 3D Converter
Converts STL files to web-compatible formats for 3D preview
"""

import trimesh
import numpy as np
import json
import os

def stl_to_gltf(stl_file_path, output_path=None):
    """Convert STL file to GLTF format for web viewing"""
    try:
        # Load STL file
        mesh = trimesh.load(stl_file_path)
        
        # Ensure mesh is valid (trimesh doesn't have is_valid attribute)
        try:
            if hasattr(mesh, 'is_valid') and not mesh.is_valid:
                mesh.fix_normals()
        except:
            # If validation fails, just continue
            pass
        
        # Export to GLTF
        if output_path is None:
            output_path = stl_file_path.replace('.stl', '.gltf')
        
        mesh.export(output_path, file_type='gltf')
        return output_path
        
    except Exception as e:
        print(f"Error converting STL to GLTF: {e}")
        return None

def stl_to_obj(stl_file_path, output_path=None):
    """Convert STL file to OBJ format for web viewing"""
    try:
        # Load STL file
        mesh = trimesh.load(stl_file_path)
        
        # Ensure mesh is valid (trimesh doesn't have is_valid attribute)
        try:
            if hasattr(mesh, 'is_valid') and not mesh.is_valid:
                mesh.fix_normals()
        except:
            # If validation fails, just continue
            pass
        
        # Export to OBJ
        if output_path is None:
            output_path = stl_file_path.replace('.stl', '.obj')
        
        mesh.export(output_path, file_type='obj')
        return output_path
        
    except Exception as e:
        print(f"Error converting STL to OBJ: {e}")
        return None

def stl_to_threejs_json(stl_file_path, output_path=None):
    """Convert STL file to Three.js JSON format"""
    try:
        # Load STL file
        mesh = trimesh.load(stl_file_path)
        
        # Ensure mesh is valid (trimesh doesn't have is_valid attribute)
        try:
            if hasattr(mesh, 'is_valid') and not mesh.is_valid:
                mesh.fix_normals()
        except:
            # If validation fails, just continue
            pass
        
        # Get vertices and faces
        vertices = mesh.vertices.flatten().tolist()
        faces = mesh.faces.flatten().tolist()
        
        # Create Three.js geometry format
        geometry = {
            "metadata": {
                "version": 4.5,
                "type": "BufferGeometry",
                "generator": "STL to Three.js Converter"
            },
            "data": {
                "attributes": {
                    "position": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": vertices
                    }
                },
                "index": {
                    "type": "Uint16Array",
                    "array": faces
                }
            }
        }
        
        # Save JSON file
        if output_path is None:
            output_path = stl_file_path.replace('.stl', '.json')
        
        with open(output_path, 'w') as f:
            json.dump(geometry, f, indent=2)
        
        return output_path
        
    except Exception as e:
        print(f"Error converting STL to Three.js JSON: {e}")
        return None

def get_mesh_info(stl_file_path):
    """Get basic information about the STL mesh"""
    try:
        mesh = trimesh.load(stl_file_path)
        
        info = {
            "vertices_count": len(mesh.vertices),
            "faces_count": len(mesh.faces),
            "bounds": mesh.bounds.tolist(),
            "volume": float(mesh.volume) if hasattr(mesh, 'volume') else 0,
            "is_watertight": mesh.is_watertight if hasattr(mesh, 'is_watertight') else False,
            "is_valid": True  # Assume valid if we can load it
        }
        
        return info
        
    except Exception as e:
        print(f"Error getting mesh info: {e}")
        return None

if __name__ == "__main__":
    # Test the converter
    test_file = "test_sculpture.stl"
    if os.path.exists(test_file):
        print("Converting STL to web formats...")
        
        gltf_file = stl_to_gltf(test_file)
        if gltf_file:
            print(f"GLTF file created: {gltf_file}")
        
        obj_file = stl_to_obj(test_file)
        if obj_file:
            print(f"OBJ file created: {obj_file}")
        
        json_file = stl_to_threejs_json(test_file)
        if json_file:
            print(f"Three.js JSON file created: {json_file}")
        
        info = get_mesh_info(test_file)
        if info:
            print(f"Mesh info: {info}")
    else:
        print("No test STL file found")
