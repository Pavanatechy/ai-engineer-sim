# AI Engineer Simulation - Quick Reference

## 🎯 What Was Built

A complete OpenEnv-compliant environment with 3 coding tasks for evaluating AI agents.

---

## 📂 Key Files at a Glance

### Core Framework

- **[env/base.py](env/base.py)** - OpenEnv models and base class (300 lines)
- **[env/registry.py](env/registry.py)** - Task registration system (50 lines)

### Three Tasks

- **[tasks/email_triage.py](tasks/email_triage.py)** - Easy: Email classification
- **[tasks/code_review.py](tasks/code_review.py)** - Medium: Bug finding
- **[tasks/data_cleaning.py](tasks/data_cleaning.py)** - Hard: Data validation

### Testing (30 tests, all passing)

- [tests/test_base.py](tests/test_base.py)
- [tests/test_email_triage.py](tests/test_email_triage.py)
- [tests/test_code_review.py](tests/test_code_review.py)
- [tests/test_data_cleaning.py](tests/test_data_cleaning.py)

### Utilities

- **[scripts/demo.py](scripts/demo.py)** - Interactive demo
- **[scripts/baseline.py](scripts/baseline.py)** - OpenAI inference script

### Documentation

- **[README.md](README.md)** - Full documentation (13KB+)
- **[BUILD_SUMMARY.md](BUILD_SUMMARY.md)** - Detailed build summary
- **[openenv.yaml](openenv.yaml)** - OpenEnv metadata
- **[Dockerfile](Dockerfile)** - Container config

---

## 🚀 Getting Started

### 1. Run the Demo (Recommended!)

```bash
python scripts/demo.py
```

Shows all 3 tasks working with real interactions.

### 2. Run All Tests

```bash
pytest tests/ -v
```

All 30 tests should pass in < 1 second.

### 3. Try a Task

```python
import tasks
from env.registry import TaskRegistry
from env.base import Action

env = TaskRegistry.instantiate("email_triage")
obs = env.reset()

action = Action(
    action_type="classify_email",
    payload={"category": "urgent"},
    reasoning="Server alert"
)

result = env.step(action)
print(result.reward.value)  # See the reward!
```

### 4. Run Baseline (requires OPENAI_API_KEY in .env)

```bash
python scripts/baseline.py --model gpt-4 --task email_triage --episodes 2
```

---

## 📊 Task Overview

### Email Triage (Easy)

- **Goal**: Classify 8 emails into 5 categories (work, meeting, urgent, policy, spam)
- **Reward**: Correct classification (+0.5), good reasoning (+0.1)
- **Max Steps**: 20
- **Data**: [data/emails.json](data/emails.json)

### Code Review (Medium)

- **Goal**: Find bugs in 5 Python code snippets and suggest fixes
- **Reward**: Bug detection (+0.3), fix quality (+0.3), reasoning (+0.2)
- **Max Steps**: 20
- **Data**: [data/code_snippets.json](data/code_snippets.json)

### Data Cleaning (Hard)

- **Goal**: Fix issues in 10-row CSV (missing values, format errors, outliers)
- **Reward**: Format fixes (+0.2), handle missing (+0.15), remove bad rows (+0.25)
- **Max Steps**: 20
- **Data**: [data/dirty_data.csv](data/dirty_data.csv)

---

## 🏗️ Architecture

```
Agent → Action → Environment → Observation
         ↓
      _grade()     (Compute reward based on correctness)
      _apply()     (Update state based on action)
      _is_done()   (Check if episode should end)
         ↓
      Reward ← returned to agent
```

Every environment:

1. Implements `_build_initial_observation()`
2. Implements `_grade(action, obs)` for scoring
3. Implements `_apply(action, obs)` for state transition
4. Implements `_is_done(obs, reward)` for episode end
5. Registered with `@TaskRegistry.register("task_id")`

---

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (optional, for baselines)
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

---

## ✅ Verification Checklist

- [x] All 3 tasks implemented and tested
- [x] 30 unit tests, 100% passing
- [x] OpenEnv interface fully compliant
- [x] Documentation complete
- [x] Docker configuration ready
- [x] Demo script working
- [x] Baseline inference script ready
- [x] Sample data for all tasks
- [x] Task registry functional
- [x] Reward function working

---

## 🎓 Understanding the Code

### Best Files to Study

1. [env/base.py](env/base.py) - Core OpenEnv implementation
2. [tasks/email_triage.py](tasks/email_triage.py) - Simplest task (good starting point)
3. [tasks/data_cleaning.py](tasks/data_cleaning.py) - Most complex task
4. [scripts/demo.py](scripts/demo.py) - Practical usage examples

### Key Classes

- `Observation` - What the agent sees
- `Action` - What the agent does
- `Reward` - Feedback signal
- `OpenEnvBase` - Base class all tasks inherit from
- `TaskRegistry` - Centralized task management

---

## 🔄 Reproducibility

All tasks use:

- Ground truth labels in data files
- Deterministic grading logic
- No randomness in validation (except seed in YAML)
- Clear success criteria

Results are reproducible across runs!

---

## 📈 Scaling

To add a new task:

1. Create `tasks/new_task.py`
2. Extend `OpenEnvBase`
3. Implement 4 abstract methods
4. Use `@TaskRegistry.register("task_id")`
5. Create `tests/test_new_task.py`
6. Update `tasks/__init__.py` imports

That's it! The registry handles the rest.

---

## 🐳 Docker

### Build

```bash
docker build -t ai-engineer-sim .
```

### Run

```bash
docker run \
    -e OPENAI_API_KEY="sk-..." \
    ai-engineer-sim \
    python scripts/baseline.py --task email_triage
```

### Deploy to HuggingFace Spaces

1. Create a Space on HuggingFace
2. Select "Docker" as runtime
3. Push this repository
4. Configure environment variables
5. Done! It auto-deploys

---

## 🐛 Debugging

### See what tasks are registered

```python
from env.registry import TaskRegistry
print(TaskRegistry.all_tasks())
```

### Check task details

```python
env = TaskRegistry.instantiate("email_triage")
obs = env.reset()
print(obs.content)
print(obs.available_actions)
print(obs.context)
```

### Validate data

```python
import json
with open("data/emails.json") as f:
    emails = json.load(f)
print(f"Emails: {len(emails)}")
for email in emails:
    print(f"  {email['id']}: {email['subject']}")
```

---

## 📞 Support

1. **Read README.md** for comprehensive docs
2. **Check tests/** for usage examples
3. **Run scripts/demo.py** to see it working
4. **Review docstrings** in task files

---

## 🎉 Summary

✅ **Complete** - All components built and tested  
✅ **Production-Ready** - 30 passing tests  
✅ **Well-Documented** - 13KB+ README  
✅ **OpenEnv Compliant** - Ready for validation  
✅ **Deployable** - Docker config included

**Time to first task**: < 1 minute (run `python scripts/demo.py`)  
**Time to run all tests**: < 1 second  
**Ready for deployment**: Yes

Enjoy your AI Engineer Simulation!
