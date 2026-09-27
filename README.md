# LoRA Fine-Tuning SmolLM2

This repository contains a Jupyter notebook for supervised fine-tuning of
[`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
with a LoRA (Low-Rank Adaptation) adapter. The training data comes from the
`everyday-conversations` configuration of the
[`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
dataset.

The notebook uses Hugging Face `transformers`, `TRL`'s `SFTTrainer`, and
`PEFT`. It trains adapter weights while keeping the base model frozen, then
includes optional adapter merging and sample text generation.

## Repository Contents

```text
.
|-- lora-finetuning.ipynb  # End-to-end fine-tuning notebook
|-- requirements.txt       # Python dependencies
`-- README.md
```

## Prerequisites

- Python 3.10 or newer
- JupyterLab, Jupyter Notebook, or Google Colab
- A Hugging Face account and access token
- A CUDA GPU with bfloat16 support is recommended for the notebook's current
  training configuration

The notebook selects CUDA, Apple Metal (MPS), or CPU automatically. However,
the supplied training arguments enable `bf16=True` and the fused AdamW
optimizer, so CPU and many MPS environments will need configuration changes
before training.

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the project dependencies and a notebook interface.

   ```powershell
   pip install -r requirements.txt
   pip install jupyterlab
   ```

3. Authenticate with Hugging Face. Run the notebook's `login()` cell and paste
   a token created at [Hugging Face Settings](https://huggingface.co/settings/tokens),
   or authenticate from the terminal:

   ```powershell
   huggingface-cli login
   ```

4. Start Jupyter and open the notebook.

   ```powershell
   jupyter lab
   ```

## Training Workflow

Run `lora-finetuning.ipynb` from top to bottom.

The current notebook configuration:

| Setting | Value |
| --- | --- |
| Base model | `HuggingFaceTB/SmolLM2-135M` |
| Dataset | `HuggingFaceTB/smoltalk` / `everyday-conversations` |
| LoRA rank | `6` |
| LoRA alpha | `8` |
| LoRA dropout | `0.05` |
| Target modules | `all-linear` |
| Epochs | `1` |
| Per-device batch size | `2` |
| Gradient accumulation | `2` |
| Learning rate | `2e-4` |
| Maximum sequence length | `1512` |
| Output directory | `SmolLM2-FT-MyDataset` |

The generated output directory contains the LoRA adapter, tokenizer files, and
training checkpoints. It is not committed to this repository.

## Customization

Edit the following cells before running training to adapt the experiment:

- Change `path` and `name` in `load_dataset(...)` to use another Hugging Face
  dataset.
- Set `finetune_name` to choose a different output directory.
- Adjust `rank_dimension`, `lora_alpha`, and `lora_dropout` to tune the LoRA
  adapter.
- Update `SFTConfig` values such as epoch count, batch size, learning rate, and
  precision for the available hardware.

For CPU or Apple Silicon runs, start by changing `bf16=False` and replacing
`optim="adamw_torch_fused"` with a supported optimizer such as
`"adamw_torch"`. Reducing batch size and sequence length can also lower memory
requirements.

## Adapter Merge and Inference

After training, the notebook includes an optional step that loads the saved
PEFT adapter, merges it into the base model with `merge_and_unload()`, and
saves the merged model to the same output directory. Run this section before
the inference section if using the provided `merged_model` generation pipeline.

The final cells format each prompt with the model's chat template and print the
generated response for several sample prompts.

## Notes

- The current implementation performs LoRA fine-tuning; despite references to
  QLoRA in notebook prose, it does not configure 4-bit model loading.
- Model and dataset downloads require network access on the first run.
- Training artifacts can be large. Keep output directories out of version
  control unless publishing a deliberately prepared model release.
