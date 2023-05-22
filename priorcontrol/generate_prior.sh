#adjust the lambda
#for std in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
#do
#python generate_prior.py --std $std --output_dir ../res/priorcontrol/generate_prior$std.txt
#done

#normal generation
python generate_prior.py --std 0.6 --output_dir ../res/priorcontrol/generate_prior.txt

#generation with extend mode
python generate_prior.py --std 0.6 --is_extend --output_dir ../res/priorcontrol/generate_prior_extend.txt