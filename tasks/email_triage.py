"""
Email Triage Task - Easy
Classify emails into categories (meeting, work, urgent, policy, spam) and prioritize them.
"""

import json
from pathlib import Path
from typing import Any

from env.base import Action, Observation, OpenEnvBase, Reward
from env.registry import TaskRegistry


@TaskRegistry.register("email_triage")
class EmailTriageEnv(OpenEnvBase):
    """
    Easy task: Classify and prioritize incoming emails.
    
    Categories: meeting, work, urgent, policy, spam
    Priority levels: 0 (urgent), 1 (high), 2 (medium), 3 (low)
    """

    MAX_STEPS = 20
    
    VALID_CATEGORIES = ["meeting", "work", "urgent", "policy", "spam"]
    VALID_ACTIONS = ["classify_email", "prioritize_email", "submit_classification"]

    def __init__(self, task_id: str = "email_triage") -> None:
        super().__init__(task_id=task_id)
        
        # Load emails from data file
        data_path = Path(__file__).parent.parent / "data" / "emails.json"
        with open(data_path) as f:
            self.emails = json.load(f)
        
        self.current_email_index = 0
        self.classifications: dict[str, str] = {}  # email_id -> category
        self.priorities: dict[str, int] = {}  # email_id -> priority
        self.score_history = []

    def _build_initial_observation(self) -> Observation:
        """Return the first email and task context."""
        if not self.emails:
            return Observation(
                task_id=self.task_id,
                step=0,
                content="No emails to process.",
                context={},
                available_actions=["submit_classification"],
            )
        
        email = self.emails[0]
        return Observation(
            task_id=self.task_id,
            step=0,
            content=f"Email {email['id']}: {email['subject']}\n\nBody: {email['body']}",
            context={
                "email_id": email["id"],
                "sender": email["sender"],
                "subject": email["subject"],
                "body": email["body"],
                "total_emails": len(self.emails),
                "processed": 0,
            },
            available_actions=self.VALID_ACTIONS,
            metadata={
                "categories": self.VALID_CATEGORIES,
                "priorities": [0, 1, 2, 3],
            },
        )

    def _grade(self, action: Action, obs: Observation) -> Reward:
        """Score the action based on correctness and reasoning."""
        email_id = obs.context.get("email_id")
        if not email_id:
            return Reward(
                value=0.0,
                cumulative=0.0,
                breakdown={},
                message="No email to grade.",
            )
        
        # Find ground truth for this email
        ground_truth = next((e for e in self.emails if e["id"] == email_id), None)
        if not ground_truth:
            return Reward(value=0.0, cumulative=0.0)
        
        breakdown = {}
        message = ""
        
        if action.action_type == "classify_email":
            predicted_category = action.payload.get("category", "")
            true_category = ground_truth["category"]
            
            correctness = 1.0 if predicted_category == true_category else -0.2
            breakdown["correctness"] = correctness
            
            # Reward reasoning
            reasoning_quality = 0.1 if action.reasoning else 0.0
            breakdown["reasoning"] = reasoning_quality
            
            value = correctness + reasoning_quality
            message = f"Classification: {predicted_category} (expected: {true_category})"
            
        elif action.action_type == "prioritize_email":
            predicted_priority = action.payload.get("priority")
            true_priority = ground_truth["priority"]
            
            # Allow ±1 on priority scale
            priority_correctness = 0.5 if abs(predicted_priority - true_priority) <= 1 else -0.1
            breakdown["priority"] = priority_correctness
            value = priority_correctness
            message = f"Priority: {predicted_priority} (expected: {true_priority})"
            
        elif action.action_type == "submit_classification":
            # Check overall accuracy
            email_count = len(self.emails)
            if self.classifications:
                correct = sum(
                    1 for email in self.emails 
                    if self.classifications.get(email["id"]) == email["category"]
                )
                accuracy = correct / email_count
                breakdown["accuracy"] = accuracy
                value = 0.5 * accuracy  # Scale to [-1, 1]
                message = f"Task submission: {correct}/{email_count} correct"
            else:
                value = -0.5
                message = "No classifications provided"
        else:
            value = 0.0
            message = f"Unknown action: {action.action_type}"
        
        return Reward(
            value=max(-1.0, min(1.0, value)),
            cumulative=0.0,  # Will be updated by framework
            breakdown=breakdown,
            message=message,
        )

    def _apply(self, action: Action, obs: Observation) -> Observation:
        """Apply the action and move to next observation."""
        email_id = obs.context.get("email_id")
        processed = obs.context.get("processed", 0)
        
        if action.action_type == "classify_email":
            self.classifications[email_id] = action.payload.get("category", "")
            
        elif action.action_type == "prioritize_email":
            self.priorities[email_id] = action.payload.get("priority", 2)
        
        # Move to next email if submitting
        if action.action_type == "submit_classification":
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content=f"Task completed. Classified {len(self.classifications)} emails.",
                context={
                    "total_emails": len(self.emails),
                    "classified": len(self.classifications),
                    "prioritized": len(self.priorities),
                },
                available_actions=["submit_classification"],
                metadata={"done": True},
            )
        
        # Move to next email
        self.current_email_index = min(self.current_email_index + 1, len(self.emails) - 1)
        if self.current_email_index < len(self.emails):
            email = self.emails[self.current_email_index]
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content=f"Email {email['id']}: {email['subject']}\n\nBody: {email['body']}",
                context={
                    "email_id": email["id"],
                    "sender": email["sender"],
                    "subject": email["subject"],
                    "body": email["body"],
                    "total_emails": len(self.emails),
                    "processed": self.current_email_index,
                },
                available_actions=self.VALID_ACTIONS,
                metadata={"categories": self.VALID_CATEGORIES},
            )
        else:
            return Observation(
                task_id=self.task_id,
                step=obs.step + 1,
                content="All emails processed. Submit your classifications.",
                context={
                    "total_emails": len(self.emails),
                    "processed": len(self.emails),
                },
                available_actions=["submit_classification"],
            )

    def _is_done(self, obs: Observation, reward: Reward) -> bool:
        """Episode ends when agent submits final classification."""
        return obs.metadata.get("done", False) or reward.message.startswith("Task completed")
