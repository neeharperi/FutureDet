python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant --jitter --K 5 
python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant --jitter --K 5 --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --jitter --K 5
python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --jitter --K 5 --cohort_analysis --eval_only 
