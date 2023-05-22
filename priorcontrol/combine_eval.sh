if [ $# == 2 ]; then
file_loc=$1
output_loc=$2
else
file_loc=../res/generate_combination.txt
output_loc=./logs/generate_combination.txt
#file_loc=../res/generate_combination_optim.txt
#output_loc=./logs/generate_combination_optim.txt
#file_loc=../res/generate_combination_optimcons.txt
#output_loc=./logs/generate_combination_optimcons.txt
fi

sh ../classify/indep_eval.sh $file_loc $output_loc
