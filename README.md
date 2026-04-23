# Ollama Model Manager (omm)

Let's be real: your time is way too valuable to read yet another README about yet another LLM tool. But since you're here—have you ever downloaded a massive 32B model, waited an hour, and then watched it painfully crawl at 2 tokens per second because it spilled out of your VRAM? Yeah, it sucks. 

`omm` exists so you can stop guessing. It looks at your Linux hardware, does the math, and finds the best Hugging Face models that will *actually* run fast on your specific machine.

## What is it?
Ollama Model Manager (`omm`) is a standalone Go CLI that bridges the gap between your local hardware resources and the Hugging Face / Ollama ecosystem. It focuses exclusively on optimizing inference performance for Linux workstations and servers.

## Key Features
* **Hardware-Aware Discovery:** Parses `/proc/meminfo` and `nvidia-smi` to establish strict parameter boundaries. If you have a dedicated NVIDIA GPU, it enforces a strict VRAM-fit heuristic to maximize throughput. If you're on an iGPU, it calculates system bandwidth-bound limits.
* **Hugging Face Integration:** Because Ollama lacks a public search API, `omm` queries the Hugging Face Hub for trending `gguf` models (like top-tier coders and sysadmin assistants) that are guaranteed to fit your hardware profile.
* **Inventory Management:** Tracks your local model library and pull progress in a clean tabular view (`omm describe`), documenting exactly *why* a model is optimal for a specific task.
* **Automation Ready:** Intended to be paired with Systemd user timers to periodically scan for new state-of-the-art models and ping your desktop (or Home Assistant webhook) when the ecosystem shifts.

## Installation
```bash
git clone https://github.com/qrkourier/ollama-model-manager.git
cd ollama-model-manager
make install
```
*(This places the `omm` binary in your `$(go env GOPATH)/bin` directory).*

## Usage
* `omm discover` - Evaluates your hardware (RAM/VRAM) and queries Hugging Face for the best matching trending models.
* `omm describe` - Prints a clean table of your managed models, their sizes, optimal tasks, and download progress.
* `omm pull` - Downloads pending models to your local Ollama instance.
