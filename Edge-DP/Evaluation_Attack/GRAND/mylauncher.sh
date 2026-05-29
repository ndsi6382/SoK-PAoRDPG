export OMP_NUM_THREADS=$(nproc --all)
export MKL_NUM_THREADS=$(nproc --all)
export OPENBLAS_NUM_THREADS=$(nproc --all)
export NUMEXPR_NUM_THREADS=$(nproc --all)

python3 benchmark_with_privgraph.py --name TmF_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../Nguyen15 - TmF/result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name PrivGraph_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../Yuan23 - PrivGraph/result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name DPGGAN_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../Yang21 - DPGGAN, DPGVAE/DPGGAN_result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name DPGVAE_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../Yang21 - DPGGAN, DPGVAE/DPGVAE_result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name DER_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../Chen14 - DER/result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name CAGMDP_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../CAGMDP, TriCycLe/CAGMDP_result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name TriCycLe_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../CAGMDP, TriCycLe/TriCycLe_result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1

python3 benchmark_with_privgraph.py --name LDPGen_eps9_i6 --dataset "../../Data/Facebook/nx_adj.pkl" --dp_graph "../../Qin17 - LDPGen/result/Facebook/nx_adj_eps9_i6.pkl" --types DDH_0.0_0.0_0_1 --graph1_props 0.0 1.0 1.0 --n_experiments 1
