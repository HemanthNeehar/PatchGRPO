---
base_model: google/gemma-4-31B-it-qat-w4a16-ct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:google/gemma-4-31B-it-qat-w4a16-ct
- lora
- transformers
- patchgrpo
- swe-bench
---

# PatchGRPO LoRA Adapter: main_lora

Official fine-tuned LoRA adapter weights for **PatchGRPO**, competing in the **Google - The Gemma 4 Developer Agent Competition**.

## Model Details
- **Base Model:** `google/gemma-4-31B-it-qat-w4a16-ct` (Compressed-Tensors INT4)
- **Adapter Type:** PEFT LoRA (Rank 16, Alpha 32)
- **Target Modules:** Query/Key/Value/Output Attention Projections + Gate/Up/Down MLP Projections (820 total tensors)
- **Optimization Strategy:** Critic-Free Group Relative Policy Optimization (GRPO) + ICML 2022 Model Souping
- **Evaluation Score:** **17.0% Pass@1** on official Kaggle hidden benchmark evaluation

## Weight Access & Downloads
Because the uncompressed `adapter_model.safetensors` file is **489.8 MB** (exceeding GitHub's 100 MB file limit), the binary weights are available via Google Drive:

- **[Download Official Checkpoints & Submissions (Google Drive)](https://drive.google.com/drive/folders/1pgYMH3i-Hp1HCKqPVkkMDwwD4_MuoH1l?usp=sharing)**
  - `submission_modelsoup_017.zip`
  - `submission_ablation_step8.zip`
  - `submission_ablation_step4.zip`
  - `main_lora_best.zip`

To reproduce the weights locally from scratch, run:
```bash
python scripts/create_model_soup.py \
    --mode dual \
    --step8_dir main_lora_online_step8-20261107T113311Z-1-001/main_lora_online_step8 \
    --best_dir main_lora_best-20261106T155831Z-1-001/main_lora_best \
    --step8_weight 0.5 \
    --best_weight 0.5 \
    --output_dir gemma-4-developer-agent/submission/adapters/main_lora
```
