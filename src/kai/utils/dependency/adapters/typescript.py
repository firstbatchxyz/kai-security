"""
TypeScript-specific domain adapter for the GraphQueryEngine.

Extends JavaScriptAdapter with TypeScript-specific patterns.
"""

from .javascript import JavaScriptAdapter


class TypeScriptAdapter(JavaScriptAdapter):
    """
    TypeScript-specific implementation of DomainAdapter.

    Extends JavaScriptAdapter since TypeScript is a superset of JavaScript.
    Adds TypeScript-specific patterns for:
    - Type annotations and interfaces
    - Decorators (e.g., NestJS)
    - Access modifiers (public, private, protected)
    """

    @property
    def name(self) -> str:
        return "typescript"

    def is_test_file(self, file_path: str) -> bool:
        """Check if a file path looks like a test file."""
        p = file_path.lower()
        return (
            "/test/" in p
            or "/tests/" in p
            or "/__tests__/" in p
            or ".test." in p
            or ".spec." in p
            or "_test." in p
            or "_spec." in p
            or "/test-" in p
            # TypeScript-specific test patterns
            or ".test.ts" in p
            or ".spec.ts" in p
            or ".test.tsx" in p
            or ".spec.tsx" in p
        )

    def is_library_file(self, file_path: str) -> bool:
        """
        Check if a file path is from an external library.

        Extends JavaScript patterns with TypeScript-specific ones.
        """
        p = file_path.lower()
        library_indicators = [
            # npm modules
            "node_modules/",
            "/node_modules/",
            # TypeScript definitions
            "@types/",
            "/@types/",
            # Declaration files (often from libraries)
            ".d.ts",
            # Common library directories
            "/vendor/",
            "/bower_components/",
            # Build output
            "/dist/",
            "/build/",
            "/out/",
            # TypeScript build cache
            "/.tsbuildinfo",
        ]
        return any(indicator in p for indicator in library_indicators)
