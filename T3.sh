python train.py --experiment FutureDetection --model pedestrian_forecast_n0
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant --extractBox --classname pedestrian --cohort_analysis
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant  --classname pedestrian --cohort_analysis --K 5 --eval_only

python train.py --experiment FutureDetection --model pedestrian_forecast_n3dtf
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --extractBox --classname pedestrian --cohort_analysis
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --classname pedestrian --cohort_analysis --K 5 --eval_only
