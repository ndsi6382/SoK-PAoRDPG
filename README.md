# Artefacts for "SoK: Practical Aspects of Releasing Differentially Private Graphs"

This repository contains the artefacts for "SoK: Practical Aspects of Releasing Differentially Private Graphs" by N. D'Silva, S. Nepal, and S. S. Kanhere (2026). The article was accepted to AsiaCCS '26, and is available [here](https://arxiv.org/abs/2603.18779). If you use this repository, please cite the following work:

```
@inproceedings{10.1145/3779208.3805972,
	author = {D'Silva, Nicholas and Nepal, Surya and Kanhere, Salil S.},
	title = {SoK: Practical Aspects of Releasing Differentially Private Graphs},
	year = {2026},
	isbn = {9798400723568},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3779208.3805972},
	doi = {10.1145/3779208.3805972},
	booktitle = {Proceedings of the ACM Asia Conference on Computer and Communications Security},
	pages = {1799–1815},
	numpages = {17},
	keywords = {graph privacy, differential privacy, graph publishing},
	location = {},
	series = {ASIA CCS '26}
}
```

## Supplementary Materials
As mentioned in the article, this repository contains all numerical and tabular results (as `.csv` files), and additional plots (as `.pdf` files) ommitted from the main article. Notably, these plots include CCDFs for the distribution-based metrics and raw measurements plotted against dataset statistics. These materials are available in the subfolders:
- Edge-DP (Example Scenario 1):
	- `/Edge-DP/Evaluation_Utility/Results/`, for evaluating Utility (Objective 4)
	- `/Edge-DP/Evaluation_Attack/`, for evaluating Empirical Privacy (Objective 3)
- Node-DP (Example Scenario 2):
	- `/Node-DP/Evaluation_Utility/Results/`, for evaluating Utility (Objective 4)
	- `/Node-DP/Evaluation_Attack/`, for evaluating Empirical Privacy (Objective 3)
- A high-resolution copy of the presented systemisation is also provided in `/Systemisation.pdf`.
- For further compatibility, Python `pickle` objects of dictionaries of the results and plots are provided within each of the subfolders mentioned above.


## Code
The `/Data/` folder contains the original datasets, along with code used for preprocessing (in Python 3 and Jupyter Notebooks).

Where necessary, further information about the code usage (including dependencies) of each method evaluated is provided in the relevant descriptively-named subfolders. Two implementations, `/Node-DP/Evaluation_Attack/Pedarsani/`, and `/Edge-DP/CAGMDP, TriCycLe/` require Java. The remaining codes use Python 3 (and Jupyter Notebooks), with the following packages *generally* required:
- `numpy`
- `networkx`
- `torch`
- `torchvision`
- `networkit`
- `cupy`
- `matplotlib`
- `tqdm`
- `scipy`
- `tensorflow`
- `scikit-learn`

The exact dependencies vary from method to method, and are provided in further detail within each subfolder.

Please note, we have reduced the file size of these artefacts with our best efforts and judgement. The artefacts contain all code, data sources, and the generated plots and numerical results. The artefacts do not contain the 5760 generated graphs and other forms of intermediate data, as the file size is infeasibly large for such a repository. Generating the required graphs for measurement and evaluation can be done so by using the provided code, however please note that the file structure has been re-organised for a smoother, more understanable perusal and observation process, rather than execution; some directory commands may need to be changed for execution.


## References
We make use of implementations from the following original sources. All credit goes to the original authors.
- Sofiane Azogagh, Zelma Aubin Birba, Josée Desharnais, Sébastien Gambs, Marc-Olivier Killijian, and Nadia Tawbi. 2025. GRAND: Graph Reconstruction from Potential Partial Adjacency and Neighborhood Data. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (Toronto ON, Canada) (KDD ’25). Association for Computing Machinery, New York, NY, USA, 47–58. doi:10.1145/3711896.373
- Xihui Chen, Sjouke Mauw, and Yunior Ramírez-Cruz. 2020. Publishing Community-Preserving Attributed Social Graphs with a Differential Privacy Guarantee. Proceedings on Privacy Enhancing Technologies 2020 (10 2020), 131–152. doi:10.2478/popets-2020-0066
- Shouling Ji, Weiqing Li, Prateek Mittal, Xin Hu, and Raheem Beyah. 2015. SecGraph: A Uniform and Open-source Evaluation System for Graph Data Anonymization and De-anonymization. In 24th USENIX Security Symposium (USENIX Security 15). USENIX Association, Washington, D.C., 303–318. https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/ji
- Shang Liu, Hao Du, Yang Cao, Bo Yan, Jinfei Liu, and Masatoshi Yoshikawa. 2025. PGB: Benchmarking Differentially Private Synthetic Graph Generation Algorithms . In 2025 IEEE 41st International Conference on Data Engineering (ICDE). IEEE Computer Society, Los Alamitos, CA, USA, 1348–1361. doi:10.1109/ICDE65448.2025.0010
- Carl Yang, Haonan Wang, Ke Zhang, Liang Chen, and Lichao Sun. 2021. Secure Deep Graph Generation with Link Differential Privacy. In The International Joint Conference on Artificial Intelligence (IJCAI).
- Quan Yuan, Zhikun Zhang, Linkang Du, Min Chen, Peng Cheng, and Mingyang Sun. 2023. PrivGraph: Differentially Private Graph Data Publication by Exploiting Community Information. In 32nd USENIX Security Symposium (USENIX Security 23). USENIX Association, Anaheim, CA, 3241–3258. https://www.usenix.org/conference/usenixsecurity23/presentation/yuan-quan
- Sen Zhang, Haibo Hu, Qingqing Ye, and Jianliang Xu. 2025. PrivDPR: Synthetic Graph Publishing with Deep PageRank under Differential Privacy. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (Toronto ON, Canada) (KDD ’25). Association for Computing Machinery, New York, NY, USA, 1936–1947. doi:10.1145/3690624.3709334

The following datasets were used:
- Jure Leskovec and Julian Mcauley. 2012. Learning to Discover Social Circles in Ego Networks. In Advances in Neural Information Processing Systems, F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger (Eds.), Vol. 25. Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2012/file/7a614fd06c325499f1680b9896beedeb-Paper.pdf
- Benedek Rozemberczki and Rik Sarkar. 2020. Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM ’20). ACM, 1325–1334.
- Benedek Rozemberczki, Carl Allen, and Rik Sarkar. 2019. Multi-scale Attributed Node Embedding. arXiv:1909.13021 [cs.LG]
- Eunjoon Cho, Seth A. Myers, and Jure Leskovec. 2011. Friendship and mobility: user movement in location-based social networks. In Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (San Diego, California, USA) (KDD ’11). Association for Computing Machinery, New York, NY, USA, 1082–1090. doi:10.1145/2020408.2020579
