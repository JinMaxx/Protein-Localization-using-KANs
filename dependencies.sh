#!/bin/bash

# Get the directory of the current script
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir" || exit
echo "Current working directory is now: $(pwd)"

# Creating virtual environment
venv="./.venv"
mkdir $venv
python -m venv $venv
python="${venv}/bin/python"   # or "pip"
pip="${venv}/bin/pip"

mkdir ./source/patches


# Installing python modules
$python $pip install --upgrade pip

# TODO: Bump up
$python $pip install -qv \
    tqdm==4.67.1 \
    h5py==3.13.0 \
    numpy==2.2.6 \
    torch==2.6.0 \
    pandas==2.2.3 \
    optuna==4.3.0 \
    plotly==5.5.0 \
    kaleido==0.2.1 \
    ipython==9.0.2 \
    pyfaidx==0.8.1.3 \
    colorcet==3.1.0 \
    biopython==1.85 \
    umap-learn==0.5.9.post2 \
    transformers==4.50.3 \
    scikit-learn==1.6.1 \
    python-dotenv==1.1.0 \
    sentencepiece==0.2.0 \
    typing-extensions==4.13.0 \
    git+https://github.com/ZiyaoLi/fast-kan.git \
    git+https://github.com/AthanasiosDelis/faster-kan.git


if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
else
  echo "Error: .env file not found. Please ensure it exists in the project root."
  exit 1
fi


echo "Setting up R environment with renv..."

# Check if renv is installed and install it if not.
Rscript -e 'if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv", repos="http://cran.us.r-project.org")'

# Initialize the renv environment if the lockfile doesn't exist yet.
if [ ! -f "renv.lock" ]; then  # # The 'init' command automatically finds dependencies and create the lockfile.
    echo "renv.lock not found. Initializing renv environment..."
    Rscript -e "renv::init(bare = TRUE, restart = FALSE)"  #  Initialize a bare environment (no prompts).
    Rscript -e "renv::snapshot(prompt = FALSE)"  # Discover dependencies and create the lockfile (no prompts).
    echo "renv environment initialized and lockfile created."
fi

# Restore the environment from the lockfile.
# This ensures that the exact package versions specified in renv.lock are installed.
echo "Restoring R packages from renv.lock..."
Rscript -e "renv::restore()"

echo "R environment setup complete."


echo "Creating project directories for model: ${ENCODING_MODEL_LOCAL}"

# Data Input and Output
mkdir -p "${ENCODINGS_INPUT_DIR_LOCAL}"
mkdir -p "${ENCODINGS_OUTPUT_DIR_LOCAL}"

# Model, Figures, and Studies
mkdir -p "${MODEL_SAVE_DIR_LOCAL}"
mkdir -p "${FIGURES_SAVE_DIR_LOCAL}"
mkdir -p "${STUDIES_SAVE_DIR_LOCAL}"

# Logging and Metrics
mkdir -p "$(dirname "${LOG_FILE_PATH_LOCAL}")"
mkdir -p "$(dirname "${TRAINING_METRICS_FILE_PATH_LOCAL}")"
mkdir -p "$(dirname "${HYPER_PARAM_METRICS_FILE_PATH_LOCAL}")"

echo "Directory setup complete."


# Just: https://github.com/casey/just
if [ ! -f "./just" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ./
fi


# Install git repositories for easier code comparison
# git clone https://github.com/HannesStark/protein-localization.git

exit 0