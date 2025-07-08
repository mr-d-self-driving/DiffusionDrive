# DiffusionDrive Notebooks

This directory contains Jupyter notebooks for analysis and visualization of DiffusionDrive experiments.

## Structure

### evaluation/
- `compare_eval.ipynb` - Compare evaluation results across different model configurations
  - Analyzes performance metrics from multiple training runs
  - Creates comparison tables for different hyperparameters (epochs, batch sizes)
  - Useful for hyperparameter tuning and model selection

- `visualization_eval.ipynb` - Visualization utilities for evaluation results
  - BEV (Bird's Eye View) plots
  - Camera view visualizations
  - Custom visualization functions

## Usage

To use these notebooks:

1. Ensure you have completed training and evaluation runs
2. Set the appropriate paths to your experiment results
3. Run the notebooks to analyze and visualize your results

## Note

These notebooks were moved from the root directory as part of the project reorganization (Community Contribution).

For NAVSIM visualization tutorials, see the `tutorial/` directory in the root of the repository.