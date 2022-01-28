python train.py --experiment FutureDetection --model pedestrian_forecast_n3dtfm
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtfm --forecast_mode velocity_dense --extractBox --classname pedestrian --cohort_analysis
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtfm --forecast_mode velocity_dense --classname pedestrian --cohort_analysis --K 5

python train.py --experiment FutureDetection --model pp_forecast_n3dtf
python evaluate.py --experiment FutureDetection --model pp_forecast_n3dtf --forecast_mode velocity_dense --extractBox --cohort_analysis
python evaluate.py --experiment FutureDetection --model pp_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --K 5

python train.py --experiment FutureDetection --model forecast_n3dtfm
python evaluate.py --experiment FutureDetection --model forecast_n3dtfm --forecast_mode velocity_dense --extractBox --cohort_analysis
python evaluate.py --experiment FutureDetection --model forecast_n3dtfm --forecast_mode velocity_dense --cohort_analysis --K 5
