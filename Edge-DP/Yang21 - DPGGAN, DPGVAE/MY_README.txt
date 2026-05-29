conda create --name dpggan2 python=3.8

conda activate dpggan2

conda install scipy matplotlib tqdm scikit-learn pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
~/miniconda3/envs/dpggan2/bin/pip install python-igraph
~/miniconda3/envs/dpggan2/bin/pip install networkx==2.5
