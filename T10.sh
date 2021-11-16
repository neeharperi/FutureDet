python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --cohort_analysis --K 5
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --cohort_analysis --K 5
python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --K 5
python evaluate.py --experiment Forecast --model forecast_n3r --forecast_mode velocity_reverse --cohort_analysis --K 5

python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --K 5
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --K 5
python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --K 5
python evaluate.py --experiment Forecast --model forecast_n3r --forecast_mode velocity_reverse --K 5



