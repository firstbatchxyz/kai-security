"""
TypeScript workspace adapter.

Extends the JavaScript workspace adapter with TypeScript-specific behavior.
In practice, TypeScript and JavaScript projects use identical workspace structures.
"""

from kai.utils.workspace.javascript import JavaScriptWorkspaceAdapter


class TypeScriptWorkspaceAdapter(JavaScriptWorkspaceAdapter):
    """
    Workspace adapter for TypeScript projects.

    Extends JavaScriptWorkspaceAdapter since TypeScript projects use
    the same workspace structure and tooling as JavaScript projects.
    """

    @property
    def framework_name(self) -> str:
        return "typescript"
