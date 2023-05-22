#combine with interpolation
python generate_combine.py --output_dir ../res/priorcontrol/generate_combination.txt

#combine with optimization without constraints
python generate_combine_optim.py --output_dir ../res/priorcontrol/generate_combination_optim.txt

#combine with optimization with constraints
python generate_combine_optim.py --is_constrained --output_dir ../res/priorcontrol/generate_combination_optimcons.txt