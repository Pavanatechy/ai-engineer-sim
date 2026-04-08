#!/usr/bin/env python3
"""
Quick start demo script showing how to use the AI Engineer Simulation environment.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tasks  # Import tasks to register them
from env.base import Action
from env.registry import TaskRegistry


def demo_email_triage():
    """Demo the email triage task."""
    print("\n" + "=" * 70)
    print("EMAIL TRIAGE TASK - Demo")
    print("=" * 70)
    
    env = TaskRegistry.instantiate("email_triage")
    obs = env.reset()
    
    print(f"\nInitial observation:\n{obs.content}\n")
    print(f"Available actions: {obs.available_actions}")
    print(f"Categories: {obs.context.get('categories', [])}")
    
    # Take first action
    action = Action(
        action_type="classify_email",
        payload={"category": "meeting"},
        reasoning="This is a scheduled team meeting"
    )
    print(f"\nAction: {action.action_type}")
    print(f"Payload: {action.payload}")
    print(f"Reasoning: {action.reasoning}")
    
    result = env.step(action)
    print(f"\nReward: {result.reward.value:.3f}")
    print(f"Message: {result.reward.message}")
    print(f"Next observation:\n{result.observation.content[:150]}...")


def demo_code_review():
    """Demo the code review task."""
    print("\n" + "=" * 70)
    print("CODE REVIEW TASK - Demo")
    print("=" * 70)
    
    env = TaskRegistry.instantiate("code_review")
    obs = env.reset()
    
    print(f"\nInitial observation:\n{obs.content}\n")
    print(f"Available actions: {obs.available_actions}")
    
    # Identify a bug
    action = Action(
        action_type="identify_bug",
        payload={"bug": "The function doesn't handle empty input"},
        reasoning="Could cause IndexError or ValueError"
    )
    print(f"\nAction: {action.action_type}")
    print(f"Payload: {action.payload}")
    print(f"Reasoning: {action.reasoning}")
    
    result = env.step(action)
    print(f"\nReward: {result.reward.value:.3f}")
    print(f"Message: {result.reward.message}")


def demo_data_cleaning():
    """Demo the data cleaning task."""
    print("\n" + "=" * 70)
    print("DATA CLEANING TASK - Demo")
    print("=" * 70)
    
    env = TaskRegistry.instantiate("data_cleaning")
    obs = env.reset()
    
    print(f"\nInitial observation:\n{obs.content}\n")
    print(f"Available actions: {obs.available_actions}")
    print(f"Data issues: {obs.context.get('issues', [])}")
    
    # Fix a format issue
    action = Action(
        action_type="fix_format",
        payload={"field": "email", "value": "john@example.com"},
        reasoning="Email was malformed, corrected format"
    )
    print(f"\nAction: {action.action_type}")
    print(f"Payload: {action.payload}")
    print(f"Reasoning: {action.reasoning}")
    
    result = env.step(action)
    print(f"\nReward: {result.reward.value:.3f}")
    print(f"Message: {result.reward.message}")


def main():
    """Run all demos."""
    print("\nAI Engineer Simulation - Quick Start Demo")
    print("=" * 70)
    print(f"\nRegistered tasks: {TaskRegistry.all_tasks()}\n")
    
    # Run demos
    demo_email_triage()
    demo_code_review()
    demo_data_cleaning()
    
    print("\n" + "=" * 70)
    print("Demo complete! Read the README.md for more information.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
