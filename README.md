## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate    # On Linux/MacOS
# .\venv\Scripts\activate   # On Windows PowerShell
Once activated, the environment name will appear at the beginning of the terminal prompt.
```

Make sure the virtual environment is active, then run:

```bash
pip install -r requirements.txt
```

This will install all the required packages listed in the `requirements.txt` file.

### Developer Setup

For development, there is an additional requirements-dev.txt file, which extends the base dependencies with tools for testing, linting, and debugging.
```bash
pip install -r requirements-dev.txt
```

This ensures you have both the core dependencies and the extra packages required for development.

### Installing PyTorch (optional with CUDA support)

PyTorch is not included in `requirements.txt` because installation depends on your system.  
You can install the CPU-only version, or (if you have a compatible NVIDIA GPU) a CUDA-enabled build.  

Check your CUDA version with:

```bash
nvcc --version
# or
nvidia-smi
```


Then follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to choose the correct command for your setup.
