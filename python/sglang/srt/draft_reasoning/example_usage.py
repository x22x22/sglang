"""
EAGLE-Think Usage Example

This example demonstrates how to integrate and use EAGLE-Think mode
with SGLang for draft reasoning capabilities.

Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0
"""

import torch
import logging
from typing import Optional

from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.draft_reasoning import EagleThinkManager, ThinkingPhase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EagleThinkExample:
    """Example class showing how to use EAGLE-Think mode."""
    
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.eagle_think_manager = EagleThinkManager(server_args)
        
        # These would be initialized with actual model workers in practice
        self.target_worker: Optional[TpModelWorker] = None
        self.draft_worker: Optional[TpModelWorker] = None
    
    def setup_workers(self, target_worker: TpModelWorker, draft_worker: TpModelWorker):
        """Setup the target and draft model workers."""
        self.target_worker = target_worker
        self.draft_worker = draft_worker
        
        # Initialize EAGLE-Think mode
        success = self.eagle_think_manager.initialize(
            target_worker=target_worker,
            draft_worker=draft_worker,
            max_thinking_length=512,  # Maximum tokens for thinking
        )
        
        if success:
            logger.info("✅ EAGLE-Think mode initialized successfully")
        else:
            logger.error("❌ Failed to initialize EAGLE-Think mode")
            
        return success
    
    def process_batch_with_thinking(self, batch: ScheduleBatch):
        """
        Process a batch with EAGLE-Think reasoning mode.
        
        This method demonstrates the main processing loop where:
        1. Normal tokens are processed by the target model
        2. <think>...</think> content is processed by the draft model
        3. Final reasoning continues with the target model
        """
        if not self.eagle_think_manager.is_enabled():
            logger.warning("EAGLE-Think mode is not enabled")
            return None
        
        try:
            # Process the batch through EAGLE-Think
            logits_output, next_token_ids, phase_changed = (
                self.eagle_think_manager.forward_batch(batch)
            )
            
            # Log phase changes for debugging
            if phase_changed:
                stats = self.eagle_think_manager.get_stats()
                current_phase = stats.get("thinking_phase", "unknown")
                logger.info(f"🔄 Phase changed to: {current_phase}")
                
                if current_phase == ThinkingPhase.THINKING.value:
                    logger.info("🧠 Entering thinking mode - draft model active")
                elif current_phase == ThinkingPhase.NORMAL.value:
                    logger.info("🎯 Returning to normal mode - target model active")
            
            return logits_output, next_token_ids
            
        except Exception as e:
            logger.error(f"Error processing batch with EAGLE-Think: {e}")
            return None
    
    def get_thinking_statistics(self) -> dict:
        """Get detailed statistics about the thinking process."""
        if not self.eagle_think_manager.is_enabled():
            return {"enabled": False, "error": "EAGLE-Think mode not enabled"}
        
        stats = self.eagle_think_manager.get_stats()
        
        # Add some additional computed statistics
        thinking_tokens_count = stats.get("thinking_tokens_count", 0)
        max_thinking_length = stats.get("max_thinking_length", 0)
        
        stats.update({
            "thinking_utilization": thinking_tokens_count / max_thinking_length if max_thinking_length > 0 else 0,
            "is_currently_thinking": stats.get("thinking_phase") == ThinkingPhase.THINKING.value,
        })
        
        return stats
    
    def reset_thinking_session(self):
        """Reset the thinking state (useful for new conversations)."""
        if self.eagle_think_manager.is_enabled():
            self.eagle_think_manager.reset_state()
            logger.info("🔄 Thinking state reset")
    
    def demonstrate_thinking_flow(self):
        """
        Demonstrate the typical flow of EAGLE-Think processing.
        
        This is a conceptual demonstration of how the system works:
        """
        logger.info("📋 EAGLE-Think Flow Demonstration:")
        logger.info("1. User input: 'Solve this math problem: 2x + 5 = 15'")
        logger.info("2. Target model generates: 'I need to solve for x. <think>'")
        logger.info("   🔄 Phase change: NORMAL → THINKING")
        logger.info("3. Draft model thinks: 'Let me work through this step by step...'")
        logger.info("   🧠 Draft model: 'First, subtract 5 from both sides: 2x = 10'")
        logger.info("   🧠 Draft model: 'Then divide by 2: x = 5'")
        logger.info("   🧠 Draft model: 'Let me verify: 2(5) + 5 = 10 + 5 = 15 ✓'")
        logger.info("4. Draft model generates: '</think>'")
        logger.info("   🔄 Phase change: THINKING → NORMAL")
        logger.info("5. Target model continues: 'The answer is x = 5.'")
        logger.info("✅ Complete response with internal reasoning!")


def create_example_server_args() -> ServerArgs:
    """Create example server arguments for EAGLE-Think."""
    # This is a simplified example - in practice, you'd load from config
    server_args = ServerArgs()
    
    # Basic configuration
    server_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    server_args.dtype = "float16"
    
    # EAGLE-Think specific settings
    server_args.enable_eagle_think = True  # Hypothetical flag
    server_args.eagle_think_max_length = 512
    
    return server_args


def main():
    """Main function demonstrating EAGLE-Think usage."""
    logger.info("🚀 Starting EAGLE-Think Example")
    
    # Create server arguments
    server_args = create_example_server_args()
    
    # Create EAGLE-Think example instance
    example = EagleThinkExample(server_args)
    
    # Demonstrate the thinking flow conceptually
    example.demonstrate_thinking_flow()
    
    # Show statistics format
    logger.info("\n📊 Example Statistics Format:")
    example_stats = {
        "enabled": True,
        "thinking_phase": "normal",
        "thinking_tokens_count": 0,
        "max_thinking_length": 512,
        "thinking_tokens": [],
        "thinking_utilization": 0.0,
        "is_currently_thinking": False,
    }
    
    for key, value in example_stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\n✨ EAGLE-Think Example Complete!")
    
    # Note about integration
    logger.info("\n📝 Integration Notes:")
    logger.info("1. Initialize EAGLE-Think with your target and draft workers")
    logger.info("2. Replace normal batch processing with eagle_think_manager.forward_batch()")
    logger.info("3. Monitor phase changes and statistics as needed")
    logger.info("4. Reset thinking state between conversations")


if __name__ == "__main__":
    main()
