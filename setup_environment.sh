#!/bin/bash
# Setup script for LLM Brain Analysis
# This script creates a conda environment with all necessary dependencies

# Set environment name
ENV_NAME="llm_neuro"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first"
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "=== Setting up environment for LLM Brain Analysis ==="

# Check if environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove it and create a new one? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME"
    else
        echo "Using existing environment. To activate it, run: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create a new conda environment with Python 3.12
echo "Creating conda environment '$ENV_NAME' with Python 3.12..."
conda create -y -n "$ENV_NAME" python=3.12
echo "Environment created"

# Activate the environment and install packages
echo "Installing required packages..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install packages from conda-forge channel
echo "Installing conda packages..."
conda install -y -c conda-forge pandas numpy matplotlib seaborn scikit-learn

# Install pip packages
echo "Installing pip packages..."
pip install openai requests python-dotenv

# Install BrainGPT dependencies (only needed if using BrainGPT model)
echo "Installing BrainGPT dependencies (peft, transformers, torch)..."
pip install peft transformers torch

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# API Keys for LLM Brain Analysis
# Replace the placeholder values with your actual API keys

OPENROUTER_API_KEY="my-token"
OPENAI_API_KEY="my-token"           # Only needed for top-functions (embeddings)
HF_TOKEN="my-token"          # Only needed for BrainGPT
EOF
    echo "Created .env template"
else
    echo ".env file already exists, skipping creation"
fi

# Create atlases directory if it doesn't exist
if [ ! -d atlases ]; then
    echo "Creating atlases directory structure..."
    mkdir -p atlases/human
    mkdir -p atlases/macaque
    mkdir -p atlases/mouse
    echo "Created atlases directory structure"
    echo "Note: You'll need to add your atlas CSV files to the appropriate species subdirectories"
else
    echo "Atlases directory already exists"
fi

# Create functions.json if it doesn't exist
if [ ! -f functions.json ]; then
    echo "Creating functions.json template..."
    cat > functions.json << 'EOF'
{
  "functions": [
    "cognitive control",
    "emotion",
    "language",
    "memory",
    "vision",
    "reward",
    "listening",
    "manipulation",
    "attention",
    "decision making",
    "cognition",
    "motor control",
    "visual perception",
    "working memory",
    "declarative memory",
    "autobiographical memory",
    "social cognition",
    "pain",
    "fixation",
    "gaze",
    "movement",
    "hearing",
    "cued attention",
    "visual attention",
    "multisensory processing",
    "action",
    "reading",
    "auditory processing",
    "inhibition"
  ],
  "groups": {
    "memory": [
      "spatial cognition",
      "rationality",
      "creativity"
    ],
    "awareness": [
      "metacognition",
      "consciousness"
    ]
  }
}
EOF
    echo "Created functions.json template"
else
    echo "functions.json already exists, skipping creation"
fi

# Provide instructions
echo ""
echo "=== Setup Complete ==="
echo "To activate the environment, run:"
echo "    conda activate $ENV_NAME"
echo ""
echo "Before running the analysis, you need to:"
echo "1. Edit .env file with your API keys (OPENROUTER_API_KEY required)"
echo "   - Get an OpenRouter key at https://openrouter.ai/keys"
echo "   - OPENAI_API_KEY only needed for top-functions (embeddings)"
echo "   - HF_TOKEN only needed for BrainGPT model"
echo "2. Add your atlas CSV files to the atlases/{species}/ directories"
echo "   Example: atlases/human/DesikanKilliany68.csv"
echo ""
echo "Then you can run the analysis with:"
echo "    python main.py functions --atlas-name DesikanKilliany68"
echo "    python main.py probabilities --atlas-name DesikanKilliany68 --functions \"memory,attention\""
echo "    python main.py test --atlas-name DesikanKilliany68"
echo ""
echo "For more options, see:"
echo "    python main.py --help"
echo ""

