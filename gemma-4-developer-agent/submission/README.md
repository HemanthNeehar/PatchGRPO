# Submission Directory Template (`submission/`)

This directory represents the canonical workspace layout for packaging Kaggle competition submissions into `submission.zip`.

```text
submission/
├── agent.yaml                  # REQUIRED: Root agent config (declares base model, instruction, tools)
├── eval_config.yaml            # Optional: Per-task evaluation budget overrides
├── configs/
│   └── sampling.yaml           # Sampling & thinking parameters (included in agent.yaml)
├── prompts/
│   ├── system.md               # System prompt for root coder agent
│   └── analyzer.md             # System prompt for code analyzer sub-agent
├── sub_agents/
│   └── code_analyzer.yaml      # Sub-agent configuration for AgentTool delegation
├── adapters/                   # Fine-tuned PEFT LoRA adapter weights (max 8 adapters, < 3 GiB)
│   ├── main_lora/
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── tool_lora/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── skills/                     # Domain-specific procedural skills
    └── repo_navigation/
        └── SKILL.md
```

## Rules for Competition Submission
1. **Single Declared Model**: All agents and sub-agents must declare `model: gemma-4-31b-it-qat-w4a16-ct` for official submission scoring.
2. **Packaging**: Create the archive directly from the contents of this folder (`zip -r submission.zip *`).
3. **No Python Code in Root**: Only declarative YAML files, markdown prompts, JSON configs, and SafeTensors weights are allowed.
