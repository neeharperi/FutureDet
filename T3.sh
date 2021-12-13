python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward --cohort_analysis --eval_only

python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --cohort_analysis --eval_only
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --K 5 --eval_only
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --cohort_analysis --K 5  --eval_only
