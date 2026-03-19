# Doctor-Patient Conversation to Clinical Note Generator

Fine-tuning Llama 3.1 8B to automatically convert doctor-patient conversation transcripts into structured clinical notes using QLoRA.

---

## Prerequisites

* **Python 3.10+**
* **Conda** (Anaconda or Miniconda) — for local environment setup
* A **Google account** (if running on Google Colab)
* A **HuggingFace account** with access to Meta Llama models (https://huggingface.co/meta-llama)
* **GPU** — this project requires a CUDA-capable GPU. An A100 is recommended, but T4/V100 will also work (slower training).

---

## Environment Setup (Local)

### Step 1: Create a Conda Environment

```bash
conda create -n clinical-note-gen python=3.10 -y
conda activate clinical-note-gen
```

### Step 2: Install PyTorch with CUDA

Install PyTorch with the appropriate CUDA version for your system. For CUDA 12.1:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

> For other CUDA versions, check https://pytorch.org/get-started/locally/

### Step 3: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install BLEURT (optional — for BLEURT evaluation metric)

```bash
pip install git+https://github.com/google-research/bleurt.git
```

### Step 5: Run the Notebook

```bash
jupyter notebook fine_tuned_LLM.ipynb
```

---

## How to Run (Google Colab — Recommended)

If you don't have a local GPU, use Google Colab instead. No environment setup is needed — the notebook installs everything automatically.

### Step 1: Open in Google Colab

Upload `fine_tuned_LLM.ipynb` to Google Colab, or open it directly if hosted on GitHub/Google Drive.

### Step 2: Select GPU Runtime

1. Go to **Runtime** > **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Select **A100** if available (under the "High-RAM" option), otherwise **T4** or **V100**
4. Click **Save**

### Step 3: Run All Cells

1. Go to **Runtime** > **Run all**
2. Alternatively, run cells one by one using **Shift + Enter**

> **Important:** After the first cell (installation cell), you may need to **restart the runtime** before continuing. Colab will prompt you if needed. After restarting, run all cells from the second cell onward.

That's it. The notebook handles everything automatically:
* Installs all dependencies
* Downloads the ACI-Bench dataset
* Loads and prepares the data
* Loads the pre-trained model
* Runs baseline evaluation
* Trains 3 hyperparameter configurations
* Evaluates and compares all models
* Performs error analysis
* Saves the best model
* Runs a demo inference

---

## Project Structure

```
.
├── fine_tuned_LLM.ipynb      # Main notebook — run this
├── requirements.txt          # Python dependencies
├── README.md                 # This file
```

### Files Generated After Running

```
.
├── aci-bench/                # Cloned dataset (created automatically)
├── best_model_lora/          # Saved best fine-tuned LoRA adapters
├── outputs_config1/          # Training checkpoints for Config 1
├── outputs_config2/          # Training checkpoints for Config 2
├── outputs_config3/          # Training checkpoints for Config 3
├── evaluation_results.pkl    # Full evaluation results (with predictions)
├── evaluation_results.json   # Metrics summary in JSON format
├── evaluation_results.png    # Visualization charts
├── sample_predictions.txt    # Example generated notes for review
```

---

## Troubleshooting

* **"CUDA out of memory"** — Restart the runtime and try again. If it persists, reduce `MAX_SEQ_LENGTH` from 4096 to 2048 in the model loading cell.
* **"Module not found" errors after installation** — Restart the runtime (or your Jupyter kernel) after the installation cell, then run from the imports cell onward.
* **Slow training** — Make sure you are using a GPU runtime. CPU-only will not work for this project.
* **Conda environment issues** — If `pip install -r requirements.txt` fails, try installing packages one at a time to identify the problematic dependency.
* **BLEURT installation fails** — BLEURT is optional. The notebook will skip BLEURT scoring if it is not installed and still run all other metrics.
