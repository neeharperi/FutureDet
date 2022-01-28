python train.py --experiment FutureDetection --model pedestrian_forecast_n3
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward  --extractBox --classname pedestrian --cohort_analysis
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward  --classname pedestrian --cohort_analysis --K 5 --eval_only

python train.py --experiment FutureDetection --model pp_pedestrian_forecast_n3dtf
python evaluate.py --experiment FutureDetection --model pp_pedestrian_forecast_n3dtf --forecast_mode velocity_dense  --extractBox --classname pedestrian --cohort_analysis
python evaluate.py --experiment FutureDetection --model pp_pedestrian_forecast_n3dtf --forecast_mode velocity_dense  --classname pedestrian --cohort_analysis --K 5 --eval_only
