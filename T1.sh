python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --K 1
python evaluate.py --experiment Forecast --model forecast_n3r --forecast_mode velocity_reverse --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_reverse --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_center --K 1
python evaluate.py --experiment Forecast --model forecast_n3d --forecast_mode velocity_dense_forward_reverse --K 1