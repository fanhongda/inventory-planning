"""
The HTTP surface.

One new component, and it is a wrapper. Every endpoint is a thin call into a function
that already exists in the package; where one could not be thin, the function was
missing from the package and was added there — `resolution.resolve_frame`,
`LandingStore.batches`, `Declarations.write_override` — rather than written here. A web
layer that computes anything computes it in a second place, and the second place is the
one the contract tests do not cover.

The rule that follows: whatever this can do, the CLI can do, producing the same
artefacts. This is a client, not a mode.
"""

from .app import create_app

__all__ = ["create_app"]
