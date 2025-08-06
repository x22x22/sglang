"""
EAGLE-Think: Draft Reasoning Mode for SGLang

This module implements a draft reasoning mode that leverages EAGLE-3's draft model
for generating <think>...</think> content, while using the target model for final reasoning.

The key idea is to use the fast draft model for internal reasoning/thinking processes
and the target model for the final output generation.

Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

logger = logging.getLogger(__name__)


class ThinkingPhase(Enum):
    """Enum to represent different phases of the thinking process."""
    NORMAL = "normal"           # Normal target model inference
    THINKING = "thinking"       # Draft model is generating <think> content
    TRANSITION = "transition"   # Transitioning from draft to target model


@dataclass
class EagleThinkInput:
    """Input data structure for EAGLE-Think mode."""
    
    # Core thinking state
    thinking_phase: ThinkingPhase = ThinkingPhase.NORMAL
    
    # Hidden states from target model (for draft model initialization)
    target_hidden_states: torch.Tensor = None
    
    # Accumulated thinking content
    thinking_tokens: List[int] = None
    thinking_hidden_states: List[torch.Tensor] = None
    
    # Configuration
    max_thinking_length: int = 512
    think_start_token_id: int = None  # <think> token ID
    think_end_token_id: int = None    # </think> token ID
    
    # Batch information
    batch_size: int = 0
    seq_lens: torch.Tensor = None
    
    def __post_init__(self):
        if self.thinking_tokens is None:
            self.thinking_tokens = []
        if self.thinking_hidden_states is None:
            self.thinking_hidden_states = []
    
    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype,
        max_thinking_length: int = 512,
        think_start_token_id: int = None,
        think_end_token_id: int = None,
    ):
        """Create an idle input for EAGLE-Think mode."""
        return cls(
            thinking_phase=ThinkingPhase.NORMAL,
            target_hidden_states=torch.empty((0, hidden_size), device=device, dtype=dtype),
            thinking_tokens=[],
            thinking_hidden_states=[],
            max_thinking_length=max_thinking_length,
            think_start_token_id=think_start_token_id,
            think_end_token_id=think_end_token_id,
            batch_size=0,
            seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
        )
    
    def should_start_thinking(self, token_id: int) -> bool:
        """Check if we should start the thinking phase."""
        return (
            self.thinking_phase == ThinkingPhase.NORMAL and 
            self.think_start_token_id is not None and
            token_id == self.think_start_token_id
        )
    
    def should_end_thinking(self, token_id: int) -> bool:
        """Check if we should end the thinking phase."""
        return (
            self.thinking_phase == ThinkingPhase.THINKING and
            self.think_end_token_id is not None and
            token_id == self.think_end_token_id
        )
    
    def should_force_end_thinking(self) -> bool:
        """Check if we should force end thinking due to length limit."""
        return (
            self.thinking_phase == ThinkingPhase.THINKING and
            len(self.thinking_tokens) >= self.max_thinking_length
        )
    
    def add_thinking_token(self, token_id: int, hidden_state: torch.Tensor):
        """Add a thinking token and its corresponding hidden state."""
        self.thinking_tokens.append(token_id)
        self.thinking_hidden_states.append(hidden_state.clone())
    
    def get_accumulated_thinking_states(self) -> Optional[torch.Tensor]:
        """Get the accumulated thinking hidden states."""
        if not self.thinking_hidden_states:
            return None
        return torch.stack(self.thinking_hidden_states, dim=0)
    
    def reset_thinking_state(self):
        """Reset the thinking state for next round."""
        self.thinking_tokens.clear()
        self.thinking_hidden_states.clear()
        self.thinking_phase = ThinkingPhase.NORMAL


@dataclass
class EagleThinkOutput:
    """Output data structure for EAGLE-Think mode."""
    
    # Generated content
    next_token_id: torch.Tensor
    logits_output: LogitsProcessorOutput
    
    # State information
    thinking_phase: ThinkingPhase
    phase_changed: bool = False
    
    # Hidden states to pass to target model (if transitioning)
    transition_hidden_states: torch.Tensor = None


class EagleThinkProcessor:
    """Main processor for EAGLE-Think mode."""
    
    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
        max_thinking_length: int = 512,
        device: torch.device = None,
    ):
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id
        self.max_thinking_length = max_thinking_length
        self.device = device or torch.device("cuda")
        
        logger.info(
            f"EagleThinkProcessor initialized with think_start_token_id={think_start_token_id}, "
            f"think_end_token_id={think_end_token_id}, max_thinking_length={max_thinking_length}"
        )
    
    def process_token_generation(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        think_input: EagleThinkInput,
    ) -> EagleThinkOutput:
        """
        Process token generation and determine the next phase.
        
        Args:
            batch: The current batch being processed
            logits_output: Output from the current model (target or draft)
            think_input: Current thinking state
            
        Returns:
            EagleThinkOutput with next token and phase information
        """
        # Sample next token from logits
        next_token_id = self._sample_token(logits_output, batch)
        phase_changed = False
        transition_hidden_states = None
        
        # Check for phase transitions
        if think_input.should_start_thinking(next_token_id.item()):
            # Transition from NORMAL to THINKING
            think_input.thinking_phase = ThinkingPhase.THINKING
            phase_changed = True
            logger.debug(f"Starting thinking phase with token {next_token_id.item()}")
            
        elif think_input.should_end_thinking(next_token_id.item()):
            # Transition from THINKING to NORMAL
            think_input.thinking_phase = ThinkingPhase.TRANSITION
            phase_changed = True
            transition_hidden_states = think_input.get_accumulated_thinking_states()
            logger.debug(f"Ending thinking phase with token {next_token_id.item()}")
            
        elif think_input.should_force_end_thinking():
            # Force end thinking due to length limit
            think_input.thinking_phase = ThinkingPhase.TRANSITION
            phase_changed = True
            transition_hidden_states = think_input.get_accumulated_thinking_states()
            logger.debug(f"Force ending thinking phase due to length limit")
        
        # If in thinking phase, accumulate the hidden states
        if think_input.thinking_phase == ThinkingPhase.THINKING:
            think_input.add_thinking_token(
                next_token_id.item(),
                logits_output.hidden_states[-1] if logits_output.hidden_states is not None else None
            )
        
        return EagleThinkOutput(
            next_token_id=next_token_id,
            logits_output=logits_output,
            thinking_phase=think_input.thinking_phase,
            phase_changed=phase_changed,
            transition_hidden_states=transition_hidden_states,
        )
    
    def _sample_token(
        self,
        logits_output: LogitsProcessorOutput,
        batch: ScheduleBatch,
    ) -> torch.Tensor:
        """Sample next token from logits output."""
        logits = logits_output.next_token_logits
        
        # Apply temperature if needed
        if hasattr(batch.sampling_info, 'temperatures') and batch.sampling_info.temperatures is not None:
            temperatures = batch.sampling_info.temperatures
            if temperatures.numel() > 0:
                logits = logits / temperatures.unsqueeze(-1)
        
        # Sample token
        if batch.sampling_info.is_all_greedy():
            next_token_id = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        return next_token_id
    
    def prepare_target_model_input(
        self,
        batch: ScheduleBatch,
        transition_hidden_states: torch.Tensor,
    ) -> ScheduleBatch:
        """
        Prepare the batch for target model inference after thinking phase.
        
        Args:
            batch: Current batch
            transition_hidden_states: Hidden states from thinking phase
            
        Returns:
            Modified batch ready for target model
        """
        # Create a new batch for target model
        target_batch = batch.copy()
        
        # Set the hidden states from thinking phase
        if transition_hidden_states is not None:
            # The target model should continue from the last thinking state
            target_batch.spec_info = None  # Clear any speculative info
            target_batch.forward_mode = ForwardMode.EXTEND
            
            # Store thinking states for target model initialization
            if not hasattr(target_batch, 'eagle_think_states'):
                target_batch.eagle_think_states = transition_hidden_states
        
        return target_batch
    
    def should_use_draft_model(self, think_input: EagleThinkInput) -> bool:
        """Determine if we should use draft model for current generation."""
        return think_input.thinking_phase == ThinkingPhase.THINKING
    
    def should_use_target_model(self, think_input: EagleThinkInput) -> bool:
        """Determine if we should use target model for current generation."""
        return think_input.thinking_phase in [ThinkingPhase.NORMAL, ThinkingPhase.TRANSITION]


def extract_multi_layer_features(
    hidden_states: torch.Tensor,
    layer_indices: List[int] = None,
) -> torch.Tensor:
    """
    Extract and fuse features from multiple layers, similar to EAGLE-3.
    
    Args:
        hidden_states: Hidden states from target model [batch_size, seq_len, hidden_size]
        layer_indices: Which layers to extract (default: [8, 16, 32] for low/mid/high)
        
    Returns:
        Fused features ready for draft model input
    """
    if layer_indices is None:
        # Default to extracting from 3 representative layers
        total_layers = hidden_states.shape[0] if len(hidden_states.shape) == 3 else 32
        layer_indices = [
            total_layers // 4,      # Low level (e.g., layer 8)
            total_layers // 2,      # Mid level (e.g., layer 16) 
            total_layers * 3 // 4,  # High level (e.g., layer 24)
        ]
    
    # Extract features from specified layers
    if len(hidden_states.shape) == 4:  # [num_layers, batch_size, seq_len, hidden_size]
        selected_features = []
        for idx in layer_indices:
            if idx < hidden_states.shape[0]:
                selected_features.append(hidden_states[idx])
        
        if selected_features:
            # Concatenate features from different layers
            fused_features = torch.cat(selected_features, dim=-1)  # [batch_size, seq_len, 3*hidden_size]
            return fused_features
    
    # Fallback: return original hidden states
    return hidden_states


def create_thinking_tokens_vocab(tokenizer) -> Tuple[int, int]:
    """
    Create or find thinking tokens in the vocabulary.
    
    Args:
        tokenizer: The tokenizer to use
        
    Returns:
        Tuple of (think_start_token_id, think_end_token_id)
    """
    # Try to find existing thinking tokens
    think_start_token = "<think>"
    think_end_token = "</think>"
    
    try:
        think_start_token_id = tokenizer.encode(think_start_token, add_special_tokens=False)[0]
        think_end_token_id = tokenizer.encode(think_end_token, add_special_tokens=False)[0]
        return think_start_token_id, think_end_token_id
    except:
        # If tokens don't exist, we might need to add them
        logger.warning(
            f"Thinking tokens {think_start_token} and {think_end_token} not found in vocabulary. "
            "Consider adding them to the tokenizer."
        )
        return None, None
