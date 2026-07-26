"""Mnemosyne — layered LLM memory subsystem.

KV cache, eviction/compression policies, tiering, prefix reuse, and the
instrumentation that attributes a measured gain to a specific mechanism.

Invariant: this package depends on torch only. It must never import proteus or
themis. The KV-cache public interface is specified in
docs/mnemosyne-cache-interface.md (authored in the Rig Design phase).
"""

__version__ = "0.1.0"
