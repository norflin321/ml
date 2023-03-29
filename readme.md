To run/train models on an Apple Silicon GPU, use the PyTorch device name "mps" with .to("mps"). [MPS](https://pytorch.org/docs/master/notes/mps.html) stands for Metal Performance Shaders, [Metal](https://developer.apple.com/metal/pytorch/) is Apple's GPU framework.

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
