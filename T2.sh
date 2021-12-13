python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --tp_pct 0.6
python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --eval_only --tp_pct 0.6

python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --eval_only --tp_pct 0.9
python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --eval_only --tp_pct 0.9

python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --eval_only --tp_pct -1
python evaluate.py --experiment Forecast --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --eval_only --tp_pct -1







