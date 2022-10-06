python generate_multi.py --config generate_config_final.json --batch_size 5 --topk 200 --strategy none --model_path ./model/AE/checkpoint-30000/pytorch_model.bin --output_dir ./res/AE/predict_final.txt --latent_size 768 --variation 1e-3
cd classify

sh indep_eval.sh ../res/AE/predict_final.txt logs/AE_final.txt
cd ..