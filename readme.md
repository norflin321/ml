<p float="left">
  <img src="https://user-images.githubusercontent.com/33498670/229221502-d72c1ead-d4f1-43c9-b1df-24b8b2077b3d.jpeg" width="49%" /> 
  <img src="https://user-images.githubusercontent.com/33498670/229221488-e26f0ba6-9c90-4aeb-9b8e-afad4c41be6a.jpg" width="49%" />
</p>

```bash
conda activate ./env

python <file>
python -m pip install <package>
conda install <library>

conda deactivate
```

<details>
  <summary>PyTorch setup for M1</summary>
  <br>

  1. [Download Miniforge3](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) (Conda installer) for macOS arm64 chips (M1, M1 Pro, M1 Max).
  2. Install Miniforge3 into home directory.
  ```bash
  chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
  sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
  source ~/miniforge3/bin/activate
  ```
  3. Restart terminal.
  4. Make and activate Conda environment. **Note:** Python 3.8 is the most stable for using the following setup.
  ```bash
  conda create --prefix ./env python=3.8
  conda activate ./env
  ```
  5. Install PyTorch.
  ```bash
  pip3 install torch torchvision torchaudio
  ```
  6. Install common data science packages.
  ```bash
  conda install pandas numpy matplotlib scikit-learn pydot pydotplus tqdm
  ```
</details>

- https://colab.research.google.com/github/ - open any notebook from github in colab select GPU and train faster
