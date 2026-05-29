python_exe=python3.9


$python_exe benchmark.py --dataset netscience --types H DDH_0.0_0.0_0_0 DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 0.1 --n_experiments 10 --log_deterministic
$python_exe benchmark.py --dataset bio-diseasome --types H DDH_0.0_0.0_0_0 DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 0.1 --n_experiments 10 --log_deterministic
$python_exe benchmark.py --dataset polblogs --types H DDH_0.0_0.0_0_0 DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 0.1 --n_experiments 10 --log_deterministic
$python_exe benchmark.py --dataset cora --types H DDH_0.0_0.0_0_0 DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 0.1 --n_experiments 10 --log_deterministic
