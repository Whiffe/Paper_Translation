"""
Core Adversarial Reasoning Algorithm Implementation
Based on Algorithm 1 from the paper
"""

import torch
import json
import re
from buffer import GWW_dfs_min
from prompts import (
    get_attacker_system_prompt,
    get_init_msg,
    get_feedbacker_system_prompt,
    get_refiner_system_prompt,
    get_judge_system_prompt
)
from utils import extract_json_field, check_jailbreak_simple


class AdversarialReasoning:
    """
    Adversarial Reasoning algorithm for jailbreaking LLMs
    """

    def __init__(self, model_wrapper, num_iterations=15, num_prompts=16,
                 num_branches=8, buffer_size=32, batch_divs=2):
        """
        Args:
            model_wrapper: Wrapper for the local model
            num_iterations: T - number of iterations
            num_prompts: n - number of attacking prompts per iteration
            num_branches: m - number of feedback branches
            buffer_size: B - buffer size for GWW
            batch_divs: k - number of divisions for feedback batching
        """
        self.model_wrapper = model_wrapper
        self.num_iterations = num_iterations
        self.num_prompts = num_prompts
        self.num_branches = num_branches
        self.buffer_size = buffer_size
        self.batch_divs = batch_divs

        # Initialize buffer
        self.buffer = GWW_dfs_min(buffer_size)

    def run(self, goal, target, task_idx=0):
        """
        Run the adversarial reasoning algorithm

        Args:
            goal: The jailbreaking goal (behavior to elicit)
            target: The target string to begin response with
            task_idx: Task index for logging

        Returns:
            Dictionary with results
        """
        print(f"Starting adversarial reasoning for task {task_idx}")
        print(f"Goal: {goal}")
        print(f"Target: {target}")

        # Initialize with root node S^(0)
        init_string = get_init_msg(goal, target)
        self.buffer = GWW_dfs_min(self.buffer_size)

        # Generate initial attacking prompts and add to buffer
        print("\n=== Iteration 0: Initialization ===")
        initial_prompts = self._generate_attacking_prompts(
            attacker_instruction=init_string,
            goal=goal,
            target=target,
            num_prompts=self.num_prompts
        )

        # Compute losses for initial prompts
        losses = self.model_wrapper.compute_loss_from_logits(initial_prompts, target)

        # Add to buffer
        self.buffer.add_prompt(init_string, losses, initial_prompts)

        print(f"Generated {len(initial_prompts)} initial prompts")
        print(f"Loss range: [{losses.min():.4f}, {losses.max():.4f}]")

        # Main loop: T iterations
        best_prompt = None
        best_response = None
        min_loss = float('inf')
        success = False

        for iteration in range(1, self.num_iterations + 1):
            print(f"\n=== Iteration {iteration}/{self.num_iterations} ===")

            # Step 1: Select node with best score from buffer
            if len(self.buffer) == 0:
                print("Buffer is empty, stopping")
                break

            current_string, current_losses, current_messages = self.buffer.get_prompt()

            if current_string is None:
                print("No more candidates in buffer")
                break

            print(f"Selected reasoning string from buffer")
            print(f"Current loss: {torch.min(torch.tensor(current_losses)):.4f}")

            # Step 2: Generate feedbacks
            feedbacks = self._generate_feedbacks(
                goal=goal,
                target=target,
                messages=current_messages,
                losses=current_losses
            )

            print(f"Generated {len(feedbacks)} feedback strings")

            # Step 3: Refine reasoning strings based on feedbacks
            new_reasoning_strings = []
            for feedback in feedbacks:
                refined_string = self._refine_reasoning_string(
                    goal=goal,
                    target=target,
                    current_string=current_string,
                    feedback=feedback
                )
                if refined_string:
                    new_reasoning_strings.append(refined_string)

            print(f"Refined into {len(new_reasoning_strings)} new reasoning strings")

            # Step 4: For each new reasoning string, generate attacking prompts
            for reasoning_string in new_reasoning_strings:
                attacking_prompts = self._generate_attacking_prompts(
                    attacker_instruction=reasoning_string,
                    goal=goal,
                    target=target,
                    num_prompts=self.num_prompts
                )

                # Compute losses
                losses = self.model_wrapper.compute_loss_from_logits(attacking_prompts, target)

                # Add to buffer
                self.buffer.add_prompt(reasoning_string, losses, attacking_prompts)

                # Track best result
                current_min_loss = torch.min(torch.tensor(losses)).item()
                if current_min_loss < min_loss:
                    min_loss = current_min_loss
                    best_idx = torch.argmin(torch.tensor(losses)).item()
                    best_prompt = attacking_prompts[best_idx]

                    # Test the prompt on target model
                    best_response = self.model_wrapper.generate("", best_prompt)

                    # Simple jailbreak check
                    if check_jailbreak_simple(best_response, target):
                        success = True
                        print(f"✓ Jailbreak successful at iteration {iteration}!")
                        print(f"Best prompt: {best_prompt[:200]}...")
                        break

            if success:
                break

            print(f"Iteration {iteration} complete. Min loss so far: {min_loss:.4f}")
            print(f"Buffer size: {len(self.buffer)}")

        # Return results
        result = {
            'success': success,
            'best_prompt': best_prompt if best_prompt else "",
            'best_response': best_response if best_response else "",
            'min_loss': min_loss,
            'iterations': iteration
        }

        return result

    def _generate_attacking_prompts(self, attacker_instruction, goal, target, num_prompts):
        """Generate attacking prompts using Attacker LLM"""
        attacker_system = get_attacker_system_prompt(goal, target)

        prompts = []
        attempts = 0
        max_attempts = num_prompts * 3  # Allow some failures

        while len(prompts) < num_prompts and attempts < max_attempts:
            attempts += 1
            try:
                response = self.model_wrapper.generate(
                    system_msg=attacker_system,
                    user_msg=attacker_instruction
                )

                # Extract "Prompt P" from JSON response
                prompt = extract_json_field(response, "Prompt P")
                if prompt and prompt not in prompts:
                    prompts.append(prompt)

            except Exception as e:
                print(f"Warning: Failed to generate attacking prompt: {e}")
                continue

        # If we don't have enough, pad with simple variations
        while len(prompts) < num_prompts:
            prompts.append(f"{goal}")

        return prompts[:num_prompts]

    def _generate_feedbacks(self, goal, target, messages, losses):
        """Generate feedback strings by comparing attacking prompts"""
        # Sort messages by loss
        sorted_indices = torch.argsort(torch.tensor(losses))
        sorted_messages = [messages[i] for i in sorted_indices]

        # Divide into k buckets and sample one from each
        batch_size = len(sorted_messages) // self.batch_divs
        sampled_messages = []
        for i in range(self.batch_divs):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i < self.batch_divs - 1 else len(sorted_messages)
            if end_idx > start_idx:
                # Sample one from this bucket (use first for determinism)
                sampled_messages.append(sorted_messages[start_idx])

        # Format feedback input
        feedback_input = self._format_feedback_input(sampled_messages)

        # Generate multiple feedbacks
        feedbacker_system = get_feedbacker_system_prompt(goal, target, len(sampled_messages))

        feedbacks = []
        for _ in range(self.num_branches):
            try:
                response = self.model_wrapper.generate(
                    system_msg=feedbacker_system,
                    user_msg=feedback_input
                )

                # Extract "Final_feedback" from JSON
                feedback = extract_json_field(response, "Final_feedback")
                if feedback:
                    feedbacks.append(feedback)

            except Exception as e:
                print(f"Warning: Failed to generate feedback: {e}")
                continue

        return feedbacks if feedbacks else ["Try a different approach."]

    def _format_feedback_input(self, messages):
        """Format messages for feedback LLM input"""
        formatted = ""
        for i, msg in enumerate(messages, 1):
            formatted += f"Prompt_{i}:\n'{msg}'\n\n"
        return formatted

    def _refine_reasoning_string(self, goal, target, current_string, feedback):
        """Refine reasoning string based on feedback"""
        refiner_system = get_refiner_system_prompt(goal, target)

        refiner_input = f"Variable_text:\n'{current_string}'\n\nFeedback:\n{feedback}"

        try:
            response = self.model_wrapper.generate(
                system_msg=refiner_system,
                user_msg=refiner_input
            )

            # Extract "Improved_variable" from JSON
            improved = extract_json_field(response, "Improved_variable")
            return improved if improved else current_string

        except Exception as e:
            print(f"Warning: Failed to refine reasoning string: {e}")
            return current_string
