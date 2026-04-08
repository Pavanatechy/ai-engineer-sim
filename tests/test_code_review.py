"""
Unit tests for CodeReviewEnv task.
"""

import pytest

from env.base import Action
from tasks.code_review import CodeReviewEnv


@pytest.fixture
def env():
    """Create a CodeReviewEnv instance."""
    return CodeReviewEnv()


def test_reset_returns_observation(env):
    """Test that reset() returns valid initial observation."""
    obs = env.reset()
    
    assert obs.task_id == "code_review"
    assert obs.step == 0
    assert "Review code snippet" in obs.content
    assert len(obs.available_actions) > 0


def test_identify_bug(env):
    """Test identifying a bug in code."""
    env.reset()
    
    action = Action(
        action_type="identify_bug",
        payload={"bug": "division by zero error"},
        reasoning="Empty list would cause division error",
    )
    
    result = env.step(action)
    assert result.reward.value >= 0  # Should reward bug identification
    assert "division" in result.reward.message.lower()


def test_suggest_fix(env):
    """Test suggesting a fix."""
    env.reset()
    
    action = Action(
        action_type="suggest_fix",
        payload={"fix": "Add a check for empty list at the start"},
        reasoning="Prevents ZeroDivisionError on empty input",
    )
    
    result = env.step(action)
    assert result.reward.value > 0  # Should reward good fixes
    assert result.observation.step == 1


def test_submit_review(env):
    """Test submitting the review."""
    env.reset()
    
    # Identify some bugs
    bug_action = Action(
        action_type="identify_bug",
        payload={"bug": "index out of bounds"},
    )
    env.step(bug_action)
    
    # Suggest a fix
    fix_action = Action(
        action_type="suggest_fix",
        payload={"fix": "Check list bounds"},
    )
    env.step(fix_action)
    
    # Submit review
    submit_action = Action(
        action_type="submit_review",
        payload={},
    )
    result = env.step(submit_action)
    assert result.done is True


def test_multiple_snippets(env):
    """Test reviewing multiple code snippets."""
    env.reset()
    initial_snippet = env._current_obs.context.get("snippet_id")
    
    # Move through snippets
    for _ in range(3):
        action = Action(
            action_type="identify_bug",
            payload={"bug": "Some bug"},
        )
        result = env.step(action)
        if result.observation.context.get("snippet_id") != initial_snippet:
            break
    
    # Should have moved to next snippet or finished
    assert env._state.step > 0


def test_step_before_reset(env):
    """Test that step() before reset() raises error."""
    action = Action(
        action_type="identify_bug",
        payload={"bug": "test"},
    )
    
    with pytest.raises(RuntimeError):
        env.step(action)


def test_invalid_action_type(env):
    """Test that invalid action types raise error."""
    env.reset()
    
    action = Action(
        action_type="invalid_review_action",
        payload={},
    )
    
    with pytest.raises(ValueError):
        env.step(action)
