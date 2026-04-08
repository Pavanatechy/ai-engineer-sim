"""
Test suite for the base OpenEnv functionality.
"""

import pytest

from env.base import Action, Observation, Reward, StepResult


def test_observation_creation():
    """Test Observation model creation."""
    obs = Observation(
        task_id="test_task",
        step=0,
        content="Test observation",
        context={"key": "value"},
        available_actions=["action1", "action2"],
    )
    assert obs.task_id == "test_task"
    assert obs.step == 0
    assert obs.content == "Test observation"
    assert obs.context == {"key": "value"}
    assert len(obs.available_actions) == 2


def test_action_creation():
    """Test Action model creation."""
    action = Action(
        action_type="test_action",
        payload={"param1": "value1"},
        reasoning="Test reasoning",
    )
    assert action.action_type == "test_action"
    assert action.payload["param1"] == "value1"
    assert action.reasoning == "Test reasoning"


def test_reward_bounds():
    """Test Reward value bounds."""
    # Valid reward
    reward = Reward(value=0.5, cumulative=0.5)
    assert -1.0 <= reward.value <= 1.0
    
    # Invalid rewards should be caught by Pydantic
    with pytest.raises(ValueError):
        Reward(value=1.5, cumulative=0.0)  # Out of bounds


def test_step_result_creation():
    """Test StepResult model creation."""
    obs = Observation(
        task_id="test",
        step=0,
        content="Test",
        available_actions=["action"],
    )
    reward = Reward(value=0.1, cumulative=0.1)
    result = StepResult(
        observation=obs,
        reward=reward,
        done=False,
    )
    assert result.observation.task_id == "test"
    assert result.reward.value == 0.1
    assert result.done is False
