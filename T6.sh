python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant 
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --cohort_analysis --eval_only 

python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward 
python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --eval_only 

python evaluate.py --experiment Forecast --model forecast_n3r --forecast_mode velocity_reverse
python evaluate.py --experiment Forecast --model forecast_n3r --forecast_mode velocity_reverse --cohort_analysis --eval_only 

python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --cohort_analysis --eval_only 

python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --eval_only --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --cohort_analysis --eval_only --K 5 


