#!/usr/bin/env python3
"""
Baseline inference script for AI Engineer Simulation environment.

This script evaluates a language model within the OpenEnv environment.
It uses the OpenAI API client to interact with a model (e.g., GPT-4).

Requirements:
- HF_TOKEN or OPENAI_API_KEY in environment variables
- Valid OpenAI/Hugging Face credentials

Usage:
    python baseline.py --model gpt-4 --task email_triage --episodes 3
    python baseline.py --model gpt-3.5-turbo --task code_review
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from env.base import Action
from env.registry import TaskRegistry

CLIENT: OpenAI | None = None


def load_env_vars():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = os.getenv("HF_TOKEN")

    if not api_key:
        raise ValueError(
            "No API credentials found. Please set OPENAI_API_KEY or HF_TOKEN in .env"
        )

    return {
        "api_key": api_key,
        "api_base_url": os.getenv("API_BASE_URL"),
        "model_name": os.getenv("MODEL_NAME"),
    }


def get_openai_client(env_vars: dict[str, str | None]) -> OpenAI:
    """Create a configured OpenAI client."""
    global CLIENT
    if CLIENT is None:
        if env_vars.get("api_base_url"):
            CLIENT = OpenAI(
                api_key=env_vars["api_key"],
                base_url=env_vars["api_base_url"],
            )
        else:
            CLIENT = OpenAI(api_key=env_vars["api_key"])
    return CLIENT


def get_model_response(model: str, system_prompt: str, observation_text: str, client: OpenAI) -> str:
    """Get response from language model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation_text},
            ],
            temperature=0.7,
            max_tokens=500,
            timeout=30,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling model: {e}")
        return ""


def get_model_response(model: str, system_prompt: str, observation_text: str) -> str:
    """Get response from language model."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": observation_text},
            ],
            temperature=0.7,
            max_tokens=500,
            timeout=30,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling model: {e}")
        return ""


def parse_model_action(model_response: str) -> Action | None:
    """Parse model response into an Action."""
    try:
        # Try to extract JSON from response
        if "{" in model_response and "}" in model_response:
            json_str = model_response[
                model_response.find("{") : model_response.rfind("}") + 1
            ]
            action_dict = json.loads(json_str)
            return Action(
                action_type=action_dict.get("action_type", "unknown"),
                payload=action_dict.get("payload", {}),
                reasoning=action_dict.get("reasoning", ""),
            )
        else:
            # Fallback: extract action type from text
            print(f"Warning: Could not parse JSON from: {model_response[:100]}")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None


def run_episode(
    task_id: str,
    model: str,
    system_prompt: str,
    max_steps: int = 20,
) -> dict:
    """Run one episode of a task with the model."""
    env = TaskRegistry.instantiate(task_id)
    obs = env.reset()
    
    episode_reward = 0.0
    rewards = []
    actions_taken = []
    
    for step in range(max_steps):
        # Prepare observation text for model
        obs_text = f"{obs.content}\n\nAvailable actions: {', '.join(obs.available_actions)}"
        
        # Get model response
        print(f"STEP {step + 1}: requesting model response")
        response = get_model_response(model, system_prompt, obs_text, client)
        print(f"STEP {step + 1}: model responded")
        print(f"  Model response: {response[:80]}...")
        
        # Parse response into action
        action = parse_model_action(response)
        if not action:
            print(f"  Warning: Could not parse action from model response")
            break
        
        actions_taken.append(
            {
                "step": step,
                "action_type": action.action_type,
                "payload": action.payload,
            }
        )
        
        try:
            # Step environment
            result = env.step(action)
            obs = result.observation
            reward = result.reward.value
            
            episode_reward += reward
            rewards.append(reward)
            
            print(
                f"    Reward: {reward:.3f}, Cumulative: {episode_reward:.3f}"
            )
            
            if result.done:
                print(f"  Episode finished at step {step + 1}")
                break
        except (ValueError, RuntimeError) as e:
            print(f"  Error during step: {e}")
            break
    
    return {
        "task_id": task_id,
        "model": model,
        "episode_reward": round(episode_reward, 4),
        "num_steps": len(actions_taken),
        "rewards": rewards,
        "actions": actions_taken,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Baseline inference for AI Engineer Simulation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use (gpt-3.5-turbo, gpt-4, etc.)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="email_triage",
        choices=["email_triage", "code_review", "data_cleaning"],
        help="Task to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_results.json",
        help="Output file for results",
    )
    
    args = parser.parse_args()
    
    # Load API credentials and client configuration
    try:
        env_vars = load_env_vars()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    client = get_openai_client(env_vars)
    
    if args.model == "gpt-3.5-turbo" and env_vars.get("model_name"):
        args.model = env_vars["model_name"]  # Use MODEL_NAME override when provided

    # Define system prompts per task
    system_prompts = {
        "email_triage": (
            "You are an expert email triage agent. Your task is to classify emails "
            "and prioritize them. For each email, respond with a JSON action like:\n"
            '{"action_type": "classify_email", "payload": {"category": "work"}, '
            '"reasoning": "Email is work-related"}'
        ),
        "code_review": (
            "You are an expert code reviewer. Your task is to identify bugs and "
            "suggest fixes in Python code. Respond with JSON actions like:\n"
            '{"action_type": "identify_bug", "payload": {"bug": "description"}, '
            '"reasoning": "why this is a bug"}'
        ),
        "data_cleaning": (
            "You are a data cleaning expert. Your task is to clean and validate "
            "CSV data. Respond with JSON actions like:\n"
            '{"action_type": "fix_format", "payload": {"field": "email", '
            '"value": "fixed@example.com"}, "reasoning": "format corrected"}'
        ),
    }
    
    # Run episodes
    results = {
        "model": args.model,
        "task": args.task,
        "episodes": args.episodes,
        "episode_results": [],
    }
    
    print("START")
    print(f"\nRunning {args.episodes} episodes of {args.task} with {args.model}")
    print("=" * 60)
    
    for i in range(args.episodes):
        print(f"\nEpisode {i + 1}/{args.episodes}")
        print("-" * 60)
        
        system_prompt = system_prompts.get(
            args.task, "You are a helpful AI assistant."
        )
        
        episode_result = run_episode(
            task_id=args.task,
            model=args.model,
            system_prompt=system_prompt,
        )
        
        results["episode_results"].append(episode_result)
        print(
            f"Episode reward: {episode_result['episode_reward']} "
            f"(steps: {episode_result['num_steps']})"
        )
    
    # Calculate summary statistics
    rewards = [
        e["episode_reward"] for e in results["episode_results"]
    ]
    results["summary"] = {
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
    }
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("END")
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Mean reward: {results['summary']['mean_reward']:.4f}")
    print(f"  Max reward: {results['summary']['max_reward']:.4f}")
    print(f"  Min reward: {results['summary']['min_reward']:.4f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
