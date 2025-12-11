"""
Buffer management using Go-with-the-Winners (GWW) algorithm
Adapted from the original implementation
"""

import torch


class GWW:
    """Basic Go-with-the-Winners buffer"""

    def __init__(self, num_prompts):
        self.num_prompts = num_prompts
        self.content = []

    def add_prompt(self, init_prompt, losses, messages):
        """Add a prompt with its losses and generated messages to buffer"""
        self.content.append({
            "init_prompt": init_prompt,
            "mean_loss": torch.mean(losses).item() if isinstance(losses, torch.Tensor) else torch.mean(torch.tensor(losses)).item(),
            "losses": losses,
            "messages": messages
        })

        self.sort_prompts()

        # Keep only top num_prompts
        if len(self.content) > self.num_prompts:
            self.content = self.content[:self.num_prompts]

    def sort_prompts(self):
        """Sort prompts by mean loss (ascending)"""
        self.content = sorted(self.content, key=lambda x: x["mean_loss"])

    def get_prompt(self):
        """Get the best prompt (lowest mean loss)"""
        if len(self.content) == 0:
            return None, None, None
        return self.content[0]["init_prompt"], self.content[0]["losses"], self.content[0]["messages"]

    def __len__(self):
        return len(self.content)


class GWW_dfs_min(GWW):
    """GWW variant that sorts by minimum loss and pops from buffer"""

    def sort_prompts(self):
        """Sort prompts by minimum loss (ascending)"""
        def get_min_loss(x):
            losses = x["losses"]
            if isinstance(losses, torch.Tensor):
                return torch.min(losses).item()
            else:
                return min(losses)

        self.content = sorted(self.content, key=get_min_loss)

    def get_prompt(self):
        """Get and remove the best prompt from buffer"""
        if len(self.content) == 0:
            return None, None, None

        first_content = self.content[0]
        self.content.pop(0)

        return first_content["init_prompt"], first_content["losses"], first_content["messages"]
