# ODSC AI East 2026 (Tutorial) - Less Compute, More Impact: How Model Quantization Fuels the Next Wave of Agentic AI

Welcome to the landing page for the session `Less Compute, More Impact: How Model Quantization Fuels the Next Wave of Agentic AI` at `ODSC AI East 2026`.

## What to Expect

This repo intends to provide an introduction to:

- Provide a guide for Fine-tuning and Quantization a Python Coding Agent
- Provide a guide for Fine-tuning and Quantization a SQL Agent
- Provide a guide for Building an Mutli-Agent (MoE) Agentic AI solution

> **IMPORTANT:** Do not actually use any of these examples models for production. There are already a **TON** of models that do this exact thing built for production. (Check out HuggingFace) Definitely, use those models over these. Some examples below...
>
> Take a look at:
> [https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
>
> CPU Quantized:
> [https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)

## Hardware Prerequisites

The SLM training and Fine-tuning projects will require a GPU (and of the H100 or better variety). There is simply no getting around that. Unfortunately, it's not going to possible to do this live in the session.

If you don't have access to this kind of hardware, you can at least download the pre-built models for inference.

## Software Prerequisites

- A Linux or Mac-based Developer’s Laptop 
  - Windows Users should use a VM or Cloud Instance
- Python Installed: version 3.12 or higher
- (Recommended) Using a miniconda or venv virtual environment
- Basic familiarity with shell operations

## Participation Options

There are 3 separate demo projects:

- [demo/1_Python](./demo/1_Python/README.md)
- [demo/2_SQL](./demo/2_SQL/README.md)
- [demo/3_A2A](./demo/3_A2A/README.md)

The instructions and purpose for each demo is contained within their respective folders.
