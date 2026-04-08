"""
Unit tests for EmailTriageEnv task.
"""

import pytest

from env.base import Action
from tasks.email_triage import EmailTriageEnv


@pytest.fixture
def env():
    """Create an EmailTriageEnv instance."""
    return EmailTriageEnv()


def test_reset_returns_observation(env):
    """Test that reset() returns valid initial observation."""
    obs = env.reset()
    
    assert obs.task_id == "email_triage"
    assert obs.step == 0
    assert obs.content != ""
    assert "email_" in obs.context.get("email_id", "")
    assert len(obs.available_actions) > 0


def test_classify_email_correct(env):
    """Test classifying an email correctly."""
    obs = env.reset()
    
    # First email is "meeting" category, so classify it correctly
    action = Action(
        action_type="classify_email",
        payload={"category": "meeting"},  # email_001 is a meeting
        reasoning="Calendar meeting scheduled",
    )
    
    result = env.step(action)
    assert result.reward.value > 0  # Should reward correct classification
    assert result.done is False


def test_classify_email_incorrect(env):
    """Test classifying an email incorrectly."""
    env.reset()
    
    action = Action(
        action_type="classify_email",
        payload={"category": "spam"},  # Wrong for email_001
        reasoning="Incorrect classification",
    )
    
    result = env.step(action)
    assert result.reward.value < 0  # Should penalize incorrect classification


def test_prioritize_email(env):
    """Test prioritizing an email."""
    env.reset()
    
    action = Action(
        action_type="prioritize_email",
        payload={"priority": 2},
        reasoning="Medium priority email",
    )
    
    result = env.step(action)
    assert result.observation.step == 1


def test_submit_classification(env):
    """Test submitting classifications."""
    env.reset()
    
    # First classify an email
    action1 = Action(
        action_type="classify_email",
        payload={"category": "meeting"},
        reasoning="Calendar related",
    )
    env.step(action1)
    
    # Then submit
    action2 = Action(
        action_type="submit_classification",
        payload={},
        reasoning="Ready to submit",
    )
    
    result = env.step(action2)
    assert result.done is True


def test_max_steps_enforced(env):
    """Test that MAX_STEPS are enforced."""
    env.reset()
    
    for _ in range(25):
        action = Action(
            action_type="classify_email",
            payload={"category": "work"},
        )
        result = env.step(action)
        if result.done:
            break
    
    # Should finish before 25 steps
    assert env._state.step <= env.MAX_STEPS


def test_invalid_action_raises_error(env):
    """Test that invalid actions raise ValueError."""
    env.reset()
    
    action = Action(
        action_type="invalid_action",
        payload={},
    )
    
    with pytest.raises(ValueError):
        env.step(action)


def test_step_before_reset_raises_error(env):
    """Test that step() before reset() raises RuntimeError."""
    action = Action(
        action_type="classify_email",
        payload={"category": "work"},
    )
    
    with pytest.raises(RuntimeError):
        env.step(action)
