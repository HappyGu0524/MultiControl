if [ $# == 2 ]; then
file_loc=$1
output_loc=$2
else
file_loc=../res/generate_prior.txt
output_loc=./logs/generate_prior.txt
#file_loc=../res/generate_prior_extend.txt
#output_loc=./logs/generate_prior_extend.txt
fi




echo 0 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [0,-1,-1] >> $output_loc

echo 1 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [1,-1,-1] >> $output_loc

echo 2 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [-1,0,-1] >> $output_loc

echo 3 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [-1,1,-1] >> $output_loc

echo 4 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [-1,2,-1] >> $output_loc

echo 5 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [-1,3,-1] >> $output_loc

echo 6 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [-1,-1,0] >> $output_loc

echo 7 >> $output_loc
python ../classify/eval.py --file_loc $file_loc --specify [-1,-1,1] >> $output_loc

