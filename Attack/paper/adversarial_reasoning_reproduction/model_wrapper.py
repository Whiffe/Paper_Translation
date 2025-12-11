"""
Model wrapper for local Qwen model
Provides unified interface for the adversarial reasoning algorithm
"""

import torch
import torch.nn as nn


class LocalModelWrapper:
    """Wrapper for local Qwen model"""

    def __init__(self, model_module, device='cuda:0'):
        """
        Args:
            model_module: The imported qwen_open_4B module
            device: Device to use ('cuda:0', 'cuda:1', 'cpu', etc.)
        """
        self.model_module = model_module
        self.device = device
        # Initialize the model
        model_module.init_qwen_open_4B()
        print(f"Qwen 4B model initialized successfully on device: {device}")
        print(f"Note: Model loading handled by qwen_open_4B module (device_map='auto')")

    def generate(self, system_msg, user_msg, enable_thinking=False):
        """
        Generate response from model

        Args:
            system_msg: System message
            user_msg: User message
            enable_thinking: Whether to enable thinking mode

        Returns:
            Generated text response
        """
        return self.model_module.call_qwen_open_4B(
            system_msg=system_msg,
            user_msg=user_msg,
            enable_thinking=enable_thinking
        )

    def generate_batch(self, system_msg, user_msgs, enable_thinking=False):
        """
        Generate responses for a batch of user messages

        Args:
            system_msg: System message (same for all)
            user_msgs: List of user messages
            enable_thinking: Whether to enable thinking mode

        Returns:
            List of generated responses
        """
        responses = []
        for user_msg in user_msgs:
            response = self.generate(system_msg, user_msg, enable_thinking)
            responses.append(response)
        return responses

    def compute_loss(self, messages, target, model, tokenizer):
        """
        Compute cross-entropy loss for target string

        Args:
            messages: List of attacking prompts
            target: Target string to match
            model: The actual model object (for logits)
            tokenizer: Tokenizer

        Returns:
            losses: Tensor of losses
        """
        # Note: For Qwen model, we need to access the underlying model
        # This is a simplified version - you may need to adapt based on actual model structure

        # For now, we'll use a proxy loss based on whether target appears in response
        losses = []
        for message in messages:
            response = self.generate("", message)
            if target.lower() in response.lower():
                loss = 0.0
            else:
                # Simple heuristic: longer responses that don't match = higher loss
                loss = 10.0 / (1.0 + len(response))
            losses.append(loss)

        return torch.tensor(losses, dtype=torch.float32)

    def compute_loss_from_logits(self, messages, target):
        """
        Compute loss using model logits (if available)
        This is a placeholder - implement if you have access to model internals
        """
        # For local Qwen model without logit access, use proxy loss
        return self.compute_loss(messages, target, None, None)
