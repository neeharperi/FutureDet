python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward --cohort_analysis --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --cohort_analysis --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_center --cohort_analysis --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --cohort_analysis --K 5

python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_center --K 5
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse