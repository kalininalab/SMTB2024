for i in 6 12 30 33; do
    for j in fluorescence stability; do
        for k in valid; do
            python preprocess.py /scratch/data/$j/raw/$k.csv $i /scratch/data/$j
        done
    done
done
