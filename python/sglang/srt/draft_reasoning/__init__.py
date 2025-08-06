"""
EAGLE-Think: Draft Reasoning Mode for SGLang

This package implements a draft reasoning mode that leverages EAGLE-3's draft model
for generating <think>...</think> content, while using the target model for final reasoning.

Main Components:
- EagleThinkProcessor: Core logic for managing thinking phases
- EagleThinkWorker: Worker that coordinates between target and draft models  
- EagleThinkManager: High-level manager for integration with SGLang

Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0
"""

from .eagle_think_utils import (
    EagleThinkInput,
    EagleThinkOutput,
    EagleThinkProcessor,
    ThinkingPhase,
    create_thinking_tokens_vocab,
    extract_multi_layer_features,
)

from .eagle_think_worker import (
    EagleThinkWorker,
    EagleThinkManager,
    draft_tp_context,
)

__all__ = [
    # Core data structures
    "EagleThinkInput",
    "EagleThinkOutput", 
    "ThinkingPhase",
    
    # Main processor and worker classes
    "EagleThinkProcessor",
    "EagleThinkWorker",
    "EagleThinkManager",
    
    # Utility functions
    "create_thinking_tokens_vocab",
    "extract_multi_layer_features",
    "draft_tp_context",
]

# Version info
__version__ = "0.1.0"
__author__ = "SGLang Team"
__description__ = "EAGLE-Think: Draft Reasoning Mode for SGLang"
