# GRAND

This code implements the graph reconstruction algorithms described in the the paper GRAND : Graph Reconstruction from Potential Partial Adjacency and Neighborhood Data (Accepted at KDD 2025).

## Requirements
- Python 
- numpy
- matplotlib
- tqdm
- prettytable
- pandas


To install the requirements, make sure to have [python](https://www.python.org) installed , run:
```bash
pip install -r requirements.txt
```

## Benchmarking the algorithms
The `benchmark.py` script runs the different attacks. Sample commands are provided in the `launcher.sh` script. To run the benchmark, execute the following command:
```bash
./launcher.sh
```

The benchmark script runs with the following parameters:

- `--dataset` : the dataset to use. The datasets are stored in the `datasets` folder.
- `--type` : the type of attack to run. The possible values are:
  - `D` : deterministic attack
  - `H` : spectral attack
  - `DDH` : deterministic attack followed by a spectral attack then a deterministic attack
- `--n_experiments` : the number of experiments to run for each attack
- `--graph1_props` : the proportion of edges in the known graph.
- `--proba_params` : the parameters of the probabilistic attack. The first, second and third parametesr are respectively alpha, beta and gamma.
- `--sanity_check_optim` : whether to run the sanity check optimization. 
- `--log_deterministic` : whether to log the deterministic attack results.

The `launcher_with_privgraph.sh` script runs the benchmark but with the private defense.
the `--epsilon` parameter is the epsilon value for the private defense.

## Results
The results are stored in the `logs` folder. The `results.ipynb` and `results_with_defense.ipynb` notebooks are used to plot the results.


## Contact
For any question, please contact us at `birba.zelma_aubin@courrier.uqam.ca` or `aubin.birba@gmail.com`.