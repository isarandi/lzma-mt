"""Autoapi extension to support .pyx (Cython) files.

This extension parses .pyx files and extracts Python-facing definitions
(def functions, classes, cdef classes) along with their docstrings and type annotations.

Usage in conf.py:
    extensions = [
        "autoapi.extension",
        "_ext.cython_autoapi",  # Must come after autoapi
    ]
    autoapi_file_patterns = ["*.py", "*.pyi", "*.pyx"]
"""

import os
import re
from pathlib import Path

import sphinx.util.logging

LOGGER = sphinx.util.logging.getLogger(__name__)

# Map Cython types to Python type annotations
CYTHON_TYPE_MAP = {
    "double": "float",
    "float": "float",
    "int": "int",
    "long": "int",
    "bint": "bool",
    "str": "str",
    "bytes": "bytes",
    "object": "Any",
    "size_t": "int",
    "Py_ssize_t": "int",
    "uint8_t": "int",
    "uint32_t": "int",
    "uint64_t": "int",
}


def parse_cython_type(cython_type: str) -> str:
    """Convert a Cython type annotation to a Python type annotation."""
    # Handle memoryviews (e.g., double[::1], float[:, :])
    if "[" in cython_type and ("::" in cython_type or ":" in cython_type):
        return "np.ndarray"

    # Handle basic types
    base_type = cython_type.split()[0]
    return CYTHON_TYPE_MAP.get(base_type, base_type)


def parse_cython_param(param: str) -> tuple[str, str | None]:
    """Parse a Cython function parameter."""
    param = param.strip()
    if not param:
        return None, None

    # Remove "not None" suffix
    param = re.sub(r"\s+not\s+None\s*$", "", param, flags=re.IGNORECASE)

    # Handle self parameter
    if param == "self":
        return "self", None

    # Handle *args and **kwargs
    if param.startswith("*"):
        return param, None

    # Pattern: type name (with optional default value)
    # e.g., "int threads=1" or "format=FORMAT_XZ"
    match = re.match(r"^(.+?)\s+(\w+)\s*(?:=.*)?$", param)
    if match:
        cython_type, name = match.groups()
        python_type = parse_cython_type(cython_type)
        return name, python_type

    # Pattern: name=default (no type)
    match = re.match(r"^(\w+)\s*=", param)
    if match:
        return match.group(1), None

    # Just a name
    if re.match(r"^\w+$", param):
        return param, None

    return None, None


def split_params(params_str: str) -> list[str]:
    """Split parameter string by commas, respecting brackets."""
    params = []
    current = []
    depth = 0
    for char in params_str:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        params.append("".join(current).strip())
    return params


def extract_docstring(text: str, after_colon: bool = True) -> str | None:
    """Extract docstring from text and dedent it properly.

    Args:
        text: The text to search for docstring
        after_colon: If True, expect docstring after a colon (for functions/classes)
    """
    if after_colon:
        pattern = r':\s*\n\s*("""|\'\'\')(.*?)\1'
    else:
        pattern = r'^\s*("""|\'\'\')(.*?)\1'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        docstring = match.group(2)
        # Dedent the docstring properly
        lines = docstring.split('\n')
        if len(lines) > 1:
            # Find minimum indentation (excluding first line and empty lines)
            min_indent = float('inf')
            for line in lines[1:]:
                stripped = line.lstrip()
                if stripped:  # Non-empty line
                    indent = len(line) - len(stripped)
                    min_indent = min(min_indent, indent)

            if min_indent < float('inf'):
                # Remove the common indentation
                dedented = [lines[0].strip()]
                for line in lines[1:]:
                    if line.strip():
                        dedented.append(line[min_indent:])
                    else:
                        dedented.append('')
                return '\n'.join(dedented).strip()
        return docstring.strip()
    return None


def parse_pyx_function(
    func_text: str,
    is_method: bool = False,
    is_staticmethod: bool = False,
    is_classmethod: bool = False,
) -> dict | None:
    """Parse a Python-facing function from .pyx file.

    Args:
        func_text: The function source text
        is_method: True if this is a method inside a class
        is_staticmethod: True if decorated with @staticmethod
        is_classmethod: True if decorated with @classmethod
    """
    pattern = r"^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:"
    match = re.match(pattern, func_text, re.MULTILINE)
    if not match:
        return None

    name = match.group(1)
    params_str = match.group(2)
    return_annotation = match.group(3)

    if return_annotation:
        return_annotation = return_annotation.strip()
        return_annotation = parse_cython_type(return_annotation)

    params = []
    if params_str.strip():
        param_list = split_params(params_str)
        # For instance methods (not static/classmethod), skip the first param (self/cls)
        # regardless of its actual name
        skip_first = is_method and not is_staticmethod
        for i, param in enumerate(param_list):
            if skip_first and i == 0:
                continue
            param_name, param_type = parse_cython_param(param)
            if param_name:
                params.append({"name": param_name, "annotation": param_type})

    docstring = extract_docstring(func_text)

    # Determine properties for the method
    properties = []
    if is_staticmethod:
        properties.append("staticmethod")
    if is_classmethod:
        properties.append("classmethod")

    return {
        "name": name,
        "params": params,
        "return_annotation": return_annotation,
        "docstring": docstring,
        "is_method": is_method,
        "properties": properties,
    }


def parse_pyx_class(class_text: str, module_name: str) -> dict | None:
    """Parse a class (regular or cdef) from .pyx file."""
    # Match both 'class ClassName:' and 'cdef class ClassName:'
    pattern = r"^(?:cdef\s+)?class\s+(\w+)(?:\s*\(([^)]*)\))?\s*:"
    match = re.match(pattern, class_text, re.MULTILINE)
    if not match:
        return None

    name = match.group(1)
    bases_str = match.group(2)

    bases = []
    if bases_str:
        bases = [b.strip() for b in bases_str.split(",") if b.strip()]

    docstring = extract_docstring(class_text)

    # Find methods within the class
    methods = []
    properties = []

    lines = class_text.split('\n')
    in_method = False
    method_lines = []
    method_indent = None
    # Track decorators for the next method
    pending_staticmethod = False
    pending_classmethod = False
    pending_property = False

    def save_method():
        """Save the current method if valid."""
        nonlocal in_method, method_lines, pending_staticmethod, pending_classmethod, pending_property
        if method_lines:
            method_text = '\n'.join(method_lines)
            method_data = parse_pyx_function(
                method_text,
                is_method=True,
                is_staticmethod=pending_staticmethod,
                is_classmethod=pending_classmethod,
            )
            if method_data and method_data.get("docstring"):
                methods.append(method_data)
        in_method = False
        method_lines = []
        # Reset decorators after using
        pending_staticmethod = False
        pending_classmethod = False
        pending_property = False

    for i, line in enumerate(lines[1:], 1):  # Skip class definition line
        stripped = line.lstrip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            if in_method:
                method_lines.append(line)
            continue

        current_indent = len(line) - len(stripped)

        # If we're in a method and hit something at method-level indent,
        # we need to end the current method first (before processing decorators)
        if in_method and current_indent <= method_indent and stripped:
            save_method()

        # Check for decorators (these come BEFORE the method they decorate)
        if stripped.startswith('@staticmethod'):
            pending_staticmethod = True
            continue
        if stripped.startswith('@classmethod'):
            pending_classmethod = True
            continue
        if stripped.startswith('@property'):
            pending_property = True
            continue
        # Skip other decorators but don't reset pending flags
        if stripped.startswith('@'):
            continue

        # Check for method definition
        if stripped.startswith('def '):
            in_method = True
            method_indent = current_indent
            method_lines = [line]
        elif in_method:
            # We're inside a method body - accumulate lines
            method_lines.append(line)

    # Don't forget the last method
    if in_method:
        save_method()

    return {
        "name": name,
        "bases": bases,
        "docstring": docstring,
        "methods": methods,
        "properties": properties,
        "full_name": f"{module_name}.{name}",
    }


def get_pyx_module_docstring(content: str) -> str | None:
    """Extract module-level docstring from .pyx content."""
    lines = content.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            start_idx = i
            break

    remaining = "\n".join(lines[start_idx:]).lstrip()
    match = re.match(r'^("""|\'\'\')(.*?)\1', remaining, re.DOTALL)
    if match:
        return match.group(2).strip()
    return None


def parse_pyx_file(filepath: str) -> dict | None:
    """Parse a .pyx file and return autoapi-compatible data structure."""
    try:
        content = Path(filepath).read_text()
    except Exception as e:
        LOGGER.warning(f"Could not read .pyx file {filepath}: {e}")
        return None

    # Determine module name from file path
    directory, filename = os.path.split(filepath)
    module_parts = []
    if filename not in ("__init__.py", "__init__.pyi", "__init__.pyx"):
        module_part = os.path.splitext(filename)[0]
        module_parts = [module_part]

    # Walk up to find package root
    while directory:
        if os.path.isfile(os.path.join(directory, "__init__.py")) or \
           os.path.isfile(os.path.join(directory, "__init__.pyi")):
            _, dir_name = os.path.split(directory)
            if dir_name:
                module_parts.insert(0, dir_name)
            directory = os.path.dirname(directory)
        else:
            break

    module_name = ".".join(module_parts)
    module_doc = get_pyx_module_docstring(content)

    children = []

    # Find all classes (both regular and cdef)
    # Pattern matches class bodies until the next class/def at the same or lower indentation
    class_pattern = r"^(?:cdef\s+)?class\s+\w+(?:\s*\([^)]*\))?\s*:.*?(?=\n(?:cdef\s+)?class\s|\ndef\s+\w+\s*\(|\Z)"
    class_matches = re.finditer(class_pattern, content, re.MULTILINE | re.DOTALL)

    for match in class_matches:
        class_text = match.group(0)
        class_data = parse_pyx_class(class_text, module_name)
        if class_data and class_data.get("docstring"):
            # Build method children
            method_children = []
            for method in class_data.get("methods", []):
                args = [
                    ("", p["name"], p["annotation"], None)
                    for p in method["params"]
                ]
                method_child = {
                    "type": "method",
                    "name": method["name"],
                    "qual_name": f"{class_data['name']}.{method['name']}",
                    "full_name": f"{module_name}.{class_data['name']}.{method['name']}",
                    "doc": method["docstring"] or "",
                    "args": args,
                    "return_annotation": method.get("return_annotation"),
                    "properties": method.get("properties", []),
                    "from_line_no": None,
                    "to_line_no": None,
                    "is_overload": False,
                    "overloads": [],
                }
                method_children.append(method_child)

            child = {
                "type": "class",
                "name": class_data["name"],
                "qual_name": class_data["name"],
                "full_name": class_data["full_name"],
                "doc": class_data["docstring"] or "",
                "bases": class_data.get("bases", []),
                "children": method_children,
                "from_line_no": None,
                "to_line_no": None,
            }
            children.append(child)

    # Find all module-level def functions (not inside classes)
    # This is trickier - we need to find 'def' at column 0
    func_pattern = r"^def\s+\w+\s*\([^)]*\)[^:]*:.*?(?=\n(?:def|cdef|class)\s|\Z)"
    func_matches = re.finditer(func_pattern, content, re.MULTILINE | re.DOTALL)

    for match in func_matches:
        func_text = match.group(0)
        func_data = parse_pyx_function(func_text)
        if func_data and func_data.get("docstring"):
            # Build args as list of tuples: (prefix, name, annotation, default)
            args = [
                ("", p["name"], p["annotation"], None)
                for p in func_data["params"]
            ]

            child = {
                "type": "function",
                "name": func_data["name"],
                "qual_name": func_data["name"],
                "full_name": f"{module_name}.{func_data['name']}",
                "doc": func_data["docstring"] or "",
                "args": args,
                "return_annotation": func_data["return_annotation"],
                "properties": [],
                "from_line_no": None,
                "to_line_no": None,
                "is_overload": False,
                "overloads": [],
            }
            children.append(child)

    if not children and not module_doc:
        return None

    return {
        "type": "module",
        "name": module_name,
        "qual_name": module_name,
        "full_name": module_name,
        "doc": module_doc or "",
        "children": children,
        "file_path": filepath,
        "encoding": "utf-8",
        "all": None,
    }


# Store reference to original Parser methods
_original_parse_file = None
_original_parse_file_in_namespace = None


def _patched_parse_file(self, file_path):
    """Patched parse_file that handles .pyx files."""
    if file_path.endswith(".pyx"):
        result = parse_pyx_file(file_path)
        if result:
            return result
        # Fall through to original if parsing failed
    return _original_parse_file(self, file_path)


def _patched_parse_file_in_namespace(self, file_path, dir_root):
    """Patched parse_file_in_namespace that handles .pyx files."""
    if file_path.endswith(".pyx"):
        result = parse_pyx_file(file_path)
        if result:
            return result
    return _original_parse_file_in_namespace(self, file_path, dir_root)


def setup(app):
    """Sphinx extension setup - patch autoapi's Parser to handle .pyx files."""
    global _original_parse_file, _original_parse_file_in_namespace

    try:
        from autoapi._parser import Parser
    except ImportError:
        LOGGER.warning("autoapi not found, cython_autoapi extension disabled")
        return {"version": "0.1", "parallel_read_safe": True}

    # Store originals and patch
    _original_parse_file = Parser.parse_file
    _original_parse_file_in_namespace = Parser.parse_file_in_namespace

    Parser.parse_file = _patched_parse_file
    Parser.parse_file_in_namespace = _patched_parse_file_in_namespace

    LOGGER.info("cython_autoapi: Patched autoapi Parser for .pyx support")

    return {"version": "0.1", "parallel_read_safe": True}
