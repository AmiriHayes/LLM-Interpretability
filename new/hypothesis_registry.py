import importlib.util
import inspect
from pathlib import Path

def hand_written_patterns():
    current_dir = Path(__file__).parent.resolve()
    # Path to your programs.py file
    programs_path = current_dir.parent / "data" / "programs.py"
    
    if not programs_path.exists():
        print(f"Error: File not found at {programs_path}")
        return []

    # 1. Create a module name for the import
    module_name = "data.programs"
    
    # 2. Load the module using spec_from_file_location
    spec = importlib.util.spec_from_file_location(module_name, programs_path)
    programs_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(programs_module)
    except Exception as e:
        print(f"Error loading programs.py: {e}")
        return []
    
    # 3. Use inspect to get functions and filter out internal Python imports
    # This ensures p.__name__ will match the 'def name()' in your file
    functions_list = [
        obj for name, obj in inspect.getmembers(programs_module) 
        if inspect.isfunction(obj) and obj.__module__ == module_name
    ]
    
    return functions_list

def greedy_patterns():
    return []

def refined_patterns():
    return []