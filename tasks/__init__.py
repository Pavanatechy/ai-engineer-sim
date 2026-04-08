"""
Task implementations for the AI Engineer Simulation environment.

Importing these modules registers the tasks with the TaskRegistry.
"""

# Import tasks to register them with TaskRegistry
from . import code_review  # noqa: F401
from . import data_cleaning  # noqa: F401
from . import email_triage  # noqa: F401

from .code_review import CodeReviewEnv
from .data_cleaning import DataCleaningEnv
from .email_triage import EmailTriageEnv

__all__ = ["EmailTriageEnv", "CodeReviewEnv", "DataCleaningEnv"]
