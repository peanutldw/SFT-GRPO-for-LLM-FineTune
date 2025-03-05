# Text-to-SQL Fine-Tuning with Unsloth

This repository implements supervised fine-tuning (SFT) and GRPO gradient optimization algorithms for large language models on Text-to-SQL tasks using the **Unsloth** efficient training framework, enabling precise conversion of natural language instructions into structured SQL queries.

## Key Features

✨ **Dual-Stage Optimization Strategy**
- **SFT (Supervised Fine-Tuning)**: Domain-specific instruction tuning with Text-to-SQL datasets
- **GRPO Optimization**: Gradient Projection-based Reinforcement Optimization algorithm to enhance SQL syntactic correctness and execution accuracy

⚡ **Extreme Training Efficiency**
- Achieves **3x training acceleration** and **70% VRAM optimization** via Unsloth
- Supports QLoRA/4-bit quantized training (70B-parameter models tunable on 24GB GPUs)

📊 **Domain-Specific Enhancements**
- Implements reward mechanisms tailored for SQL syntactic structures
- Incorporates Text-to-SQL specialization strategies including query rewriting and schema linking

### Project-Specific Dependencies
```bash
pip install vllm unsloth
