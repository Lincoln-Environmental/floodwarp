import ast
import importlib.util
from pathlib import Path

import floodwarp


def test_runtime_import_graph_contains_only_standalone_modules():
    directory = Path(floodwarp.__file__).parent
    excluded = {'overland_flow', 'floodfm_data', 'DarcyWarp', 'RichardsWarp', 'SHANZ',
                'cnn_surrogate', 'ensemble_generator', 'research', 'tools'}
    for path in directory.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [entry.name for entry in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module]
            for name in imports:
                assert name.split('.')[0] not in excluded, (path.name, name)
                if name.startswith('floodwarp.'):
                    assert importlib.util.find_spec(name) is not None
    for name in ('kinematic_solver', 'semi_implicit_solver', 'inclined_inertial_solver',
                 'regional_routing', 'nested_simulation'):
        assert not (directory/(name+'.py')).exists()
