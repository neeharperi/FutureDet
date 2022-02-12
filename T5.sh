python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --cohort_analysis
python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --K 5 --cohort_analysis --eval_only 
python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --K 0 --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis 
python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --K 5 --cohort_analysis --eval_only 
python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --K 0 --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant  --cohort_analysis --classname pedestrian
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant  --K 5 --cohort_analysis --eval_only --classname pedestrian
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant  --K 0 --cohort_analysis --eval_only --classname pedestrian

python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward --cohort_analysis --classname pedestrian
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward --K 5 --cohort_analysis --eval_only --classname pedestrian
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward --K 0 --cohort_analysis --eval_only --classname pedestrian