#!/usr/bin/env python3
"""
Demo script showing how to use the HTTP client to interact with AI Engineer Simulation.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AIEngineerEnv
from env.base import Action


def demo_client():
    """Demonstrate the HTTP client connecting to a running server."""
    print("🤖 AI Engineer Simulation - HTTP Client Demo")
    print("=" * 50)

    # Connect to the server
    client = AIEngineerEnv(base_url="http://localhost:8000", task_id="email_triage")

    try:
        print("🔄 Resetting environment...")
        result = client.reset()
        print("✅ Reset successful!")
        print(f"   Task: {result.observation.task_id}")
        print(f"   Step: {result.observation.step}")
        print(f"   Available actions: {result.observation.available_actions}")
        print(f"   Content preview: {result.observation.content[:100]}...")

        # Take a few steps
        for step in range(3):
            print(f"\n🔄 Step {step + 1}")
            # Use the first available action with sample payload
            action_type = result.observation.available_actions[0]
            action = Action(
                action_type=action_type,
                payload={"category": "urgent", "reasoning": "Demo action"}
            )

            result = client.step(action)
            print(f"   Action: {action_type}")
            print(f"   Reward: {result.reward.value:.2f}")
            print(f"   Done: {result.done}")
            print(f"   Content preview: {result.observation.content[:50]}...")

            if result.done:
                break

        # Get final state
        print("\n📊 Final state:")
        state = client.state()
        print(f"   Episode: {state.episode_id}")
        print(f"   Total steps: {state.step}")
        print(f"   Cumulative reward: {state.cumulative_reward:.2f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the server is running: python -m uvicorn inference:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    demo_client()