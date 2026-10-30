"""Reference-profile construction for ReproBench.

The reference profiles are the *ground truth* side of the benchmark: they
describe, per CVE, the firmware artifacts, binary paths, endpoints/parameters
and trigger-evidence type that a successful reproduction should surface.
Profiles are built offline-first from the curated dataset and content-addressed
source cache, verified, and are **never** mounted into the agent container.

Public entry points
-------------------
``discover_sources`` / ``SourceSpec``
    Turn a ``data/cves.yaml`` entry into an ordered list of evidence sources.
``fetch_sources``
    Fetch (or replay from cache) those sources.
``clean_document`` / ``chunk_document``
    Readability-style HTML -> text conversion and <=32KB anchored chunks.
``build_profile``
    Distil fetched evidence into ``profile.json`` + ``info.txt``.
``verify_profile``
    Enforce the Table 6 verification gate and set
    ``concrete_trigger_evidence_verified``.
"""

from __future__ import annotations

from .clean import Chunk, chunk_document, chunk_text, clean_document, html_to_text
from .distill import build_profile, build_profiles, load_profile, profile_path
from .fetch import FetchedSource, fetch_sources
from .sources import SourceSpec, discover_sources, source_priority
from .verify import verify_profile, verification_rate

__all__ = [
    "Chunk",
    "FetchedSource",
    "SourceSpec",
    "build_profile",
    "build_profiles",
    "chunk_document",
    "chunk_text",
    "clean_document",
    "discover_sources",
    "fetch_sources",
    "html_to_text",
    "load_profile",
    "profile_path",
    "source_priority",
    "verification_rate",
    "verify_profile",
]
