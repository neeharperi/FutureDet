python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --nogroup
python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --K 0 --cohort_analysis --eval_only  --nogroup

#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --classname pedestrian
#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --K 5 --cohort_analysis --eval_only --classname pedestrian
#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --K 0 --cohort_analysis --eval_only --classname pedestrian
