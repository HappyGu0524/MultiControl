if [ $# == 2 ]; then
file_loc=$1
output_loc=$2
else
file_loc=../res/AE/predict.txt
output_loc=logs/indep_AE.txt
fi

for i in 0 1
do
    for j in 0 1 2 3
    do
        echo [$i,$j,1] >> $output_loc
        python eval.py --file_loc $file_loc --specify [$i,$j,1] >> $output_loc
    done;
done;
echo average >> $output_loc
python eval.py --file_loc $file_loc >> $output_loc