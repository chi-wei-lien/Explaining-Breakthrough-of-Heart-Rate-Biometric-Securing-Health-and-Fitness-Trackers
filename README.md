### Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Link to Paper](#link-to-paper)

# Introduction

This project marks the beginning of my research journey, which I started after my second semester at Purdue. At the time, my knowledge of machine learning was limited. However, under the guidance of Professor Vhaduri, I gained a lot of valuable experience throughout this project. While I don't consider it groundbreaking, I wanted to share it with you.

In this project, we developed an authentication model that verifies a user’s identity using their heart rate data (measured in beats per minute). We used a feature we refer to as the eigenheart to perform the verification task. The file `train.py` contains the code snippet for the training process.

# Tech Stack

- Python
- Scikit-learn
- Matplotlib
- Pandas

# How to Run

1. Install the required dependencies. It is recommended to install them in a virtual environment.
   ```sh
   pip install -r requirements.txt
   ```
2. Run `train.py`. If you're using a virtual environment, make sure to activate it first.
   ```sh
   python train.py
   ```
3. After running the script, you should see the following three files generated:
   - `best_params.xlsx`: Contains the best hyperparameters found using hyperparameter optimization with 5-fold cross validation.
   - `confusion_matrix.xlsx`: Contains the results of the confusion matrix.
   - `summary.xlsx`: Contains the training summary.

# Link to Paper

Lien, Chi-Wei, et al. "Explaining vulnerabilities of heart rate biometric models securing IoT wearables." Machine Learning with Applications (2024): 100559. [link](https://doi.org/10.1016/j.mlwa.2024.100559)
