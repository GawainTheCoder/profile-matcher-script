"""
LinkedIn profile matching from Upwork data.

This package provides modular components for matching Upwork freelancer profiles
to LinkedIn profiles using SERP search providers and optional LLM assistance.
"""

__version__ = "2.0.0"

# Main exports
from .providers import SearchProvider, SerpAPIProvider, SerperProvider
from .models import RowFeatures, CandidateEvidence, infer_gl
from .features import prepare_row_features
from .queries import build_queries
from .scoring import score_candidate
from .llm import generate_llm_queries, llm_rerank_candidates
from .utils import (
    norm, truncate_text, canonicalize_linkedin_url,
    is_linkedin_profile, pick_confidence
)

__all__ = [
    # Providers
    'SearchProvider', 'SerpAPIProvider', 'SerperProvider',
    # Models
    'RowFeatures', 'CandidateEvidence', 'infer_gl',
    # Features
    'prepare_row_features',
    # Queries
    'build_queries',
    # Scoring
    'score_candidate',
    # LLM
    'generate_llm_queries', 'llm_rerank_candidates',
    # Utils
    'norm', 'truncate_text', 'canonicalize_linkedin_url',
    'is_linkedin_profile', 'pick_confidence',
]