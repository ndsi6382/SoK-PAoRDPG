export OMP_NUM_THREADS=$(nproc --all)
export MKL_NUM_THREADS=$(nproc --all)
export OPENBLAS_NUM_THREADS=$(nproc --all)
export NUMEXPR_NUM_THREADS=$(nproc --all)

for e in 0.5 0.75 1 1.5 2 3 4.5 6.5 9 12 16 20; do
    for i in 9 8 7 6 5 4 3 2 1 0; do
        nice -n 10 python3 privcom_cupy.py GitHub $e $i
    done
done
