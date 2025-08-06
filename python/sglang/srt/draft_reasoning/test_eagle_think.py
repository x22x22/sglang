"""
Unit tests for EAGLE-Think: Draft Reasoning Mode

This module contains comprehensive unit tests for the EAGLE-Think implementation,
covering all major components and functionality.

Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0
"""

import unittest
from unittest.mock import Mock, patch
import torch
import logging

from sglang.srt.draft_reasoning.eagle_think_utils import (
    EagleThinkInput,
    EagleThinkProcessor,
    ThinkingPhase,
    create_thinking_tokens_vocab,
    extract_multi_layer_features,
)

from sglang.srt.draft_reasoning.eagle_think_worker import (
    EagleThinkWorker,
    EagleThinkManager,
)

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestEagleThinkInput(unittest.TestCase):
    """Test cases for EagleThinkInput data structure."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 4096
        self.dtype = torch.float16
        
    def test_create_idle_input(self):
        """Test creating idle input."""
        think_input = EagleThinkInput.create_idle_input(
            device=self.device,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            max_thinking_length=512,
            think_start_token_id=1000,  # Use smaller token ID
            think_end_token_id=1001,    # Use smaller token ID
        )
        
        self.assertEqual(think_input.thinking_phase, ThinkingPhase.NORMAL)
        self.assertEqual(think_input.max_thinking_length, 512)
        self.assertEqual(think_input.think_start_token_id, 1000)
        self.assertEqual(think_input.think_end_token_id, 1001)
        self.assertEqual(len(think_input.thinking_tokens), 0)
        self.assertEqual(len(think_input.thinking_hidden_states), 0)
    
    def test_should_start_thinking(self):
        """Test thinking start detection."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.NORMAL,
            think_start_token_id=1000,
            think_end_token_id=1001,
        )
        
        # Should start thinking when encountering start token
        self.assertTrue(think_input.should_start_thinking(1000))
        self.assertFalse(think_input.should_start_thinking(12345))
        
        # Should not start thinking if already thinking
        think_input.thinking_phase = ThinkingPhase.THINKING
        self.assertFalse(think_input.should_start_thinking(1000))
    
    def test_should_end_thinking(self):
        """Test thinking end detection."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.THINKING,
            think_start_token_id=1000,
            think_end_token_id=1001,
        )
        
        # Should end thinking when encountering end token
        self.assertTrue(think_input.should_end_thinking(1001))
        self.assertFalse(think_input.should_end_thinking(12345))
        
        # Should not end thinking if not currently thinking
        think_input.thinking_phase = ThinkingPhase.NORMAL
        self.assertFalse(think_input.should_end_thinking(1001))
    
    def test_should_force_end_thinking(self):
        """Test force ending thinking due to length limit."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.THINKING,
            max_thinking_length=3,
            thinking_tokens=[1, 2, 3],  # At limit
            think_start_token_id=1000,
            think_end_token_id=1001,
        )
        
        self.assertTrue(think_input.should_force_end_thinking())
        
        # Should not force end if under limit
        think_input.thinking_tokens = [1, 2]
        self.assertFalse(think_input.should_force_end_thinking())
        
        # Should not force end if not thinking
        think_input.thinking_phase = ThinkingPhase.NORMAL
        think_input.thinking_tokens = [1, 2, 3, 4]  # Over limit but not thinking
        self.assertFalse(think_input.should_force_end_thinking())
    
    def test_add_thinking_token(self):
        """Test adding thinking tokens and hidden states."""
        think_input = EagleThinkInput()
        hidden_state = torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype)
        
        think_input.add_thinking_token(12345, hidden_state)
        
        self.assertEqual(len(think_input.thinking_tokens), 1)
        self.assertEqual(think_input.thinking_tokens[0], 12345)
        self.assertEqual(len(think_input.thinking_hidden_states), 1)
        self.assertTrue(torch.equal(think_input.thinking_hidden_states[0], hidden_state))
    
    def test_get_accumulated_thinking_states(self):
        """Test getting accumulated thinking states."""
        think_input = EagleThinkInput()
        
        # Empty case
        self.assertIsNone(think_input.get_accumulated_thinking_states())
        
        # Add some states
        hidden_states = [
            torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype),
            torch.randn(1, self.hidden_size, device=self.device, dtype=self.dtype),
        ]
        
        for i, hs in enumerate(hidden_states):
            think_input.add_thinking_token(i, hs)
        
        accumulated = think_input.get_accumulated_thinking_states()
        self.assertEqual(accumulated.shape, (2, 1, self.hidden_size))
    
    def test_reset_thinking_state(self):
        """Test resetting thinking state."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.THINKING,
            thinking_tokens=[1, 2, 3],
        )
        
        think_input.reset_thinking_state()
        
        self.assertEqual(think_input.thinking_phase, ThinkingPhase.NORMAL)
        self.assertEqual(len(think_input.thinking_tokens), 0)
        self.assertEqual(len(think_input.thinking_hidden_states), 0)


class TestEagleThinkProcessor(unittest.TestCase):
    """Test cases for EagleThinkProcessor."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = 50000  # Define vocab size
        self.processor = EagleThinkProcessor(
            think_start_token_id=1000,  # Use smaller token ID within vocab
            think_end_token_id=1001,    # Use smaller token ID within vocab
            max_thinking_length=512,
            device=self.device,
        )
        
        # Mock objects
        self.mock_batch = Mock(spec=ScheduleBatch)
        self.mock_batch.sampling_info = Mock()
        self.mock_batch.sampling_info.is_all_greedy.return_value = True
        self.mock_batch.sampling_info.temperatures = torch.tensor([1.0], device=self.device)
        
        self.mock_logits_output = Mock(spec=LogitsProcessorOutput)
        self.mock_logits_output.next_token_logits = torch.randn(1, self.vocab_size, device=self.device)
        self.mock_logits_output.hidden_states = torch.randn(1, 4096, device=self.device)
    
    def test_process_token_generation_normal_phase(self):
        """Test processing tokens in normal phase."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.NORMAL,
            think_start_token_id=1000,
            think_end_token_id=1001,
        )
        
        # Test normal token (not start token)
        self.mock_logits_output.next_token_logits = torch.zeros(1, self.vocab_size, device=self.device)
        self.mock_logits_output.next_token_logits[0, 12345] = 10.0  # Make token 12345 most likely
        
        result = self.processor.process_token_generation(
            self.mock_batch, self.mock_logits_output, think_input
        )
        
        self.assertEqual(result.thinking_phase, ThinkingPhase.NORMAL)
        self.assertFalse(result.phase_changed)
        self.assertEqual(result.next_token_id.item(), 12345)
    
    def test_process_token_generation_start_thinking(self):
        """Test starting thinking phase."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.NORMAL,
            think_start_token_id=1000,
            think_end_token_id=1001,
        )
        
        # Make start token most likely
        self.mock_logits_output.next_token_logits = torch.zeros(1, self.vocab_size, device=self.device)
        self.mock_logits_output.next_token_logits[0, 1000] = 10.0
        
        result = self.processor.process_token_generation(
            self.mock_batch, self.mock_logits_output, think_input
        )
        
        self.assertEqual(result.thinking_phase, ThinkingPhase.THINKING)
        self.assertTrue(result.phase_changed)
        self.assertEqual(result.next_token_id.item(), 1000)
    
    def test_process_token_generation_end_thinking(self):
        """Test ending thinking phase."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.THINKING,
            think_start_token_id=1000,
            think_end_token_id=1001,
        )
        
        # Add some thinking tokens and hidden states to simulate thinking process
        dummy_hidden_state = torch.randn(1, 4096, device=self.device)
        think_input.add_thinking_token(500, dummy_hidden_state)
        think_input.add_thinking_token(600, dummy_hidden_state)
        
        # Make end token most likely
        self.mock_logits_output.next_token_logits = torch.zeros(1, self.vocab_size, device=self.device)
        self.mock_logits_output.next_token_logits[0, 1001] = 10.0
        
        result = self.processor.process_token_generation(
            self.mock_batch, self.mock_logits_output, think_input
        )
        
        self.assertEqual(result.thinking_phase, ThinkingPhase.TRANSITION)
        self.assertTrue(result.phase_changed)
        self.assertEqual(result.next_token_id.item(), 1001)
        self.assertIsNotNone(result.transition_hidden_states)
    
    def test_process_token_generation_force_end_thinking(self):
        """Test force ending thinking due to length limit."""
        think_input = EagleThinkInput(
            thinking_phase=ThinkingPhase.THINKING,
            think_start_token_id=1000,
            think_end_token_id=1001,
            max_thinking_length=2,
            thinking_tokens=[1, 2],  # At limit
        )
        
        # Add corresponding hidden states for the thinking tokens
        dummy_hidden_state = torch.randn(1, 4096, device=self.device)
        think_input.thinking_hidden_states = [dummy_hidden_state.clone(), dummy_hidden_state.clone()]
        
        # Any token should trigger force end
        self.mock_logits_output.next_token_logits = torch.zeros(1, self.vocab_size, device=self.device)
        self.mock_logits_output.next_token_logits[0, 12345] = 10.0
        
        result = self.processor.process_token_generation(
            self.mock_batch, self.mock_logits_output, think_input
        )
        
        self.assertEqual(result.thinking_phase, ThinkingPhase.TRANSITION)
        self.assertTrue(result.phase_changed)
    
    def test_should_use_draft_model(self):
        """Test draft model usage decision."""
        think_input = EagleThinkInput(thinking_phase=ThinkingPhase.THINKING)
        self.assertTrue(self.processor.should_use_draft_model(think_input))
        
        think_input.thinking_phase = ThinkingPhase.NORMAL
        self.assertFalse(self.processor.should_use_draft_model(think_input))
    
    def test_should_use_target_model(self):
        """Test target model usage decision."""
        think_input = EagleThinkInput(thinking_phase=ThinkingPhase.NORMAL)
        self.assertTrue(self.processor.should_use_target_model(think_input))
        
        think_input.thinking_phase = ThinkingPhase.TRANSITION
        self.assertTrue(self.processor.should_use_target_model(think_input))
        
        think_input.thinking_phase = ThinkingPhase.THINKING
        self.assertFalse(self.processor.should_use_target_model(think_input))


class TestExtractMultiLayerFeatures(unittest.TestCase):
    """Test cases for multi-layer feature extraction."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_extract_multi_layer_features_4d(self):
        """Test extracting features from 4D tensor (with layer dimension)."""
        # [num_layers, batch_size, seq_len, hidden_size]
        hidden_states = torch.randn(32, 2, 10, 4096, device=self.device)
        
        fused_features = extract_multi_layer_features(hidden_states)
        
        # Should concatenate features from 3 layers
        expected_shape = (2, 10, 4096 * 3)  # [batch_size, seq_len, 3*hidden_size]
        self.assertEqual(fused_features.shape, expected_shape)
    
    def test_extract_multi_layer_features_3d_fallback(self):
        """Test fallback for 3D tensor."""
        # [batch_size, seq_len, hidden_size]
        hidden_states = torch.randn(2, 10, 4096, device=self.device)
        
        fused_features = extract_multi_layer_features(hidden_states)
        
        # Should return original tensor as fallback
        self.assertTrue(torch.equal(fused_features, hidden_states))
    
    def test_extract_multi_layer_features_custom_indices(self):
        """Test extracting features with custom layer indices."""
        hidden_states = torch.randn(32, 2, 10, 4096, device=self.device)
        layer_indices = [0, 15, 31]  # First, middle, last layers
        
        fused_features = extract_multi_layer_features(hidden_states, layer_indices)
        
        expected_shape = (2, 10, 4096 * 3)
        self.assertEqual(fused_features.shape, expected_shape)


class TestCreateThinkingTokensVocab(unittest.TestCase):
    """Test cases for thinking tokens vocabulary creation."""
    
    def test_create_thinking_tokens_vocab_success(self):
        """Test successful creation of thinking tokens."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            "<think>": [1000],
            "</think>": [1001],
        }[text]
        
        start_id, end_id = create_thinking_tokens_vocab(mock_tokenizer)
        
        self.assertEqual(start_id, 1000)
        self.assertEqual(end_id, 1001)
    
    def test_create_thinking_tokens_vocab_failure(self):
        """Test handling of tokenizer failure."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        start_id, end_id = create_thinking_tokens_vocab(mock_tokenizer)
        
        self.assertIsNone(start_id)
        self.assertIsNone(end_id)


class TestEagleThinkWorker(unittest.TestCase):
    """Test cases for EagleThinkWorker."""
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mock server args
        self.mock_server_args = Mock(spec=ServerArgs)
        self.mock_server_args.device = self.device
        
        # Mock workers
        self.mock_target_worker = Mock()
        self.mock_target_worker.model_runner.model_config.hidden_size = 4096
        self.mock_target_worker.model_runner.model_config.dtype = torch.float16
        self.mock_target_worker.tokenizer = Mock()
        self.mock_target_worker.tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            "<think>": [1000],
            "</think>": [1001],
        }[text]
        
        self.mock_draft_worker = Mock()
        self.mock_draft_worker.model_runner.model_config.hidden_size = 4096
        self.mock_draft_worker.model_runner.model_config.dtype = torch.float16
        self.mock_draft_worker.model_runner.tp_group = Mock()
    
    @patch('sglang.srt.draft_reasoning.eagle_think_worker.draft_tp_context')
    def test_initialization(self, mock_draft_tp_context):
        """Test EagleThinkWorker initialization."""
        worker = EagleThinkWorker(
            server_args=self.mock_server_args,
            target_worker=self.mock_target_worker,
            draft_worker=self.mock_draft_worker,
            max_thinking_length=512,
        )
        
        self.assertEqual(worker.max_thinking_length, 512)
        self.assertEqual(worker.think_start_token_id, 1000)
        self.assertEqual(worker.think_end_token_id, 1001)
        self.assertIsNotNone(worker.processor)
        self.assertIsNotNone(worker.current_think_input)
    
    def test_is_thinking_mode_enabled(self):
        """Test checking if thinking mode is enabled."""
        worker = EagleThinkWorker(
            server_args=self.mock_server_args,
            target_worker=self.mock_target_worker,
            draft_worker=self.mock_draft_worker,
        )
        
        self.assertTrue(worker.is_thinking_mode_enabled())
        
        # Test disabled case
        worker.think_start_token_id = None
        self.assertFalse(worker.is_thinking_mode_enabled())
    
    def test_get_thinking_stats(self):
        """Test getting thinking statistics."""
        worker = EagleThinkWorker(
            server_args=self.mock_server_args,
            target_worker=self.mock_target_worker,
            draft_worker=self.mock_draft_worker,
        )
        
        stats = worker.get_thinking_stats()
        
        self.assertIn("thinking_phase", stats)
        self.assertIn("thinking_tokens_count", stats)
        self.assertIn("max_thinking_length", stats)
        self.assertIn("thinking_tokens", stats)
    
    def test_reset_thinking_state(self):
        """Test resetting thinking state."""
        worker = EagleThinkWorker(
            server_args=self.mock_server_args,
            target_worker=self.mock_target_worker,
            draft_worker=self.mock_draft_worker,
        )
        
        # Add some thinking tokens
        worker.current_think_input.thinking_phase = ThinkingPhase.THINKING
        worker.current_think_input.thinking_tokens = [1, 2, 3]
        
        worker.reset_thinking_state()
        
        self.assertEqual(worker.current_think_input.thinking_phase, ThinkingPhase.NORMAL)
        self.assertEqual(len(worker.current_think_input.thinking_tokens), 0)
    
    def test_set_thinking_length_limit(self):
        """Test updating thinking length limit."""
        worker = EagleThinkWorker(
            server_args=self.mock_server_args,
            target_worker=self.mock_target_worker,
            draft_worker=self.mock_draft_worker,
            max_thinking_length=512,
        )
        
        worker.set_thinking_length_limit(1024)
        
        self.assertEqual(worker.max_thinking_length, 1024)
        self.assertEqual(worker.current_think_input.max_thinking_length, 1024)
        self.assertEqual(worker.processor.max_thinking_length, 1024)


class TestEagleThinkManager(unittest.TestCase):
    """Test cases for EagleThinkManager."""
    
    def setUp(self):
        self.mock_server_args = Mock(spec=ServerArgs)
        self.mock_server_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.manager = EagleThinkManager(self.mock_server_args)
    
    def test_initialization(self):
        """Test EagleThinkManager initialization."""
        self.assertIsNotNone(self.manager.server_args)
        self.assertIsNone(self.manager.eagle_think_worker)
        self.assertFalse(self.manager.enabled)
    
    def test_initialize_success(self):
        """Test successful initialization with workers."""
        mock_target_worker = Mock()
        mock_target_worker.model_runner.model_config.hidden_size = 4096
        mock_target_worker.model_runner.model_config.dtype = torch.float16
        mock_target_worker.tokenizer = Mock()
        mock_target_worker.tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
            "<think>": [1000],
            "</think>": [1001],
        }[text]
        
        mock_draft_worker = Mock()
        mock_draft_worker.model_runner.model_config.hidden_size = 4096
        mock_draft_worker.model_runner.model_config.dtype = torch.float16
        mock_draft_worker.model_runner.tp_group = Mock()
        
        success = self.manager.initialize(
            target_worker=mock_target_worker,
            draft_worker=mock_draft_worker,
        )
        
        self.assertTrue(success)
        self.assertTrue(self.manager.enabled)
        self.assertIsNotNone(self.manager.eagle_think_worker)
    
    def test_initialize_no_draft_worker(self):
        """Test initialization without draft worker."""
        mock_target_worker = Mock()
        
        success = self.manager.initialize(
            target_worker=mock_target_worker,
            draft_worker=None,
        )
        
        self.assertFalse(success)
        self.assertFalse(self.manager.enabled)
    
    def test_is_enabled(self):
        """Test checking if manager is enabled."""
        self.assertFalse(self.manager.is_enabled())
        
        # Mock successful initialization
        self.manager.enabled = True
        self.manager.eagle_think_worker = Mock()
        
        self.assertTrue(self.manager.is_enabled())
    
    def test_get_stats_disabled(self):
        """Test getting stats when disabled."""
        stats = self.manager.get_stats()
        
        self.assertFalse(stats["enabled"])
    
    def test_get_stats_enabled(self):
        """Test getting stats when enabled."""
        # Mock enabled state
        self.manager.enabled = True
        mock_worker = Mock()
        mock_worker.get_thinking_stats.return_value = {
            "thinking_phase": "normal",
            "thinking_tokens_count": 0,
        }
        self.manager.eagle_think_worker = mock_worker
        
        stats = self.manager.get_stats()
        
        self.assertTrue(stats["enabled"])
        self.assertIn("thinking_phase", stats)


def run_tests():
    """Run all EAGLE-Think tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestEagleThinkInput,
        TestEagleThinkProcessor,
        TestExtractMultiLayerFeatures,
        TestCreateThinkingTokensVocab,
        TestEagleThinkWorker,
        TestEagleThinkManager,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running EAGLE-Think unit tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
