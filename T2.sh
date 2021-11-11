python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --K 1
python evaluate.py --experiment Forecast --model forecast_n3r --forecast_mode velocity_reverse --cohort_analysis --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward --cohort_analysis --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --cohort_analysis --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_center --cohort_analysis --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --cohort_analysis --K 1