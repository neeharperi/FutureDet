python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward --cohort_analysis
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --cohort_analysis
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_center --cohort_analysis
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --cohort_analysis

python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_center
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse

