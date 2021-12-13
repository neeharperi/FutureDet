python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --tp_pct 0.6
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --eval_only --tp_pct 0.6
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --cohort_analysis --eval_only --tp_pct 0.6
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --cohort_analysis --eval_only --tp_pct 0.6

python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --eval_only --tp_pct 0.9
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --eval_only --tp_pct 0.9
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --cohort_analysis --eval_only --tp_pct 0.9
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --cohort_analysis --eval_only --tp_pct 0.9

python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --eval_only --tp_pct -1
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --eval_only --tp_pct -1
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --cohort_analysis --eval_only --tp_pct 01
python evaluate.py --experiment Forecast --model forecast_n0 --forecast_mode velocity_constant --static_only --cohort_analysis --eval_only --tp_pct -1



