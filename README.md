# Distributed-Naive-Bayes-Classifier

## Overview
This project implements a Naive Bayes Classifier from scratch for categorical datasets, using various approaches including single-threaded, multi-threaded, and distributed computing. The classifier predicts class labels based on input features and supports efficient parallel computation.

## Key Features:

### File 1: Multi-threaded Naive Bayes Classifier
Uses Python's multiprocessing and threading libraries to distribute computation across multiple threads and processes.
Demonstrates parameter estimation using counters and logical operations for efficient computations.

### File 2: Distributed Naive Bayes Classifier (MPI)
Leverages mpi4py for distributed computation.
Splits data across multiple processes to compute probabilities independently and aggregates the results using MPI communication.

### File 3: Single-threaded Naive Bayes Classifier
A basic implementation of the Naive Bayes algorithm.
Provides a clean and simple approach to training and predicting class labels for categorical datasets.

## Installation Steps

1. Install necessary packages

    ```sudo apt-get -y update && sudo apt-get -y install python3 git python3-pip python3-venv libmpich-dev```

2. Create virtual environment

    ```python3 -m venv /venv```

3. Install pip packages

    ```/venv/bin/pip install numpy pandas mpi4py```

4. Clone the repository

    ```git clone https://github.com/AvinashSubhash/Distributed-Naive-Bayes-Classifier.git /NBC```

5. Run the bash script

    ```./Output.sh```
