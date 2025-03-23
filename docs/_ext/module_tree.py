from docutils import nodes
from sphinx.util.docutils import SphinxDirective
import importlib
import pkgutil
import inspect

class ModuleTree(SphinxDirective):
    has_content = True
    required_arguments = 1

    def _build_tree(self, module_name, level=0):
        try:
            module = importlib.import_module(module_name)
            result = []

            # Only show the leaf name for deeper levels
            display_name = module_name.split('.')[-1] if level > 0 else module_name

            if level == 0:
                result.append((display_name + ':', level))
            else:
                result.append((display_name + ':', level))

            # Get submodules
            submodules = []
            for _, name, ispkg in pkgutil.iter_modules(module.__path__, module_name + '.'):
                submodules.append(name)

            # Sort submodules
            submodules.sort()

            # Process each submodule
            for submod_name in submodules:
                result.extend(self._build_tree(submod_name, level + 1))

            return result
        except (ImportError, AttributeError):
            return [(f"{module_name.split('.')[-1]}", level)]

    def run(self):
        module_name = self.arguments[0]
        tree_items = self._build_tree(module_name)

        paragraph = nodes.paragraph()
        for text, level in tree_items:
            line = nodes.line()
            line += nodes.Text('    ' * level + text)
            paragraph += line

        return [paragraph]

def setup(app):
    app.add_directive('module_tree', ModuleTree)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
    }
