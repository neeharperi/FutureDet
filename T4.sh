python train.py --experiment FutureDetection --model forecast_n3dtfm
python evaluate.py --experiment FutureDetection --model forecast_n3dtfm --forecast_mode velocity_dense --extractBox --cohort_analysis
python evaluate.py --experiment FutureDetection --model forecast_n3dtfm --forecast_mode velocity_dense --cohort_analysis --K 5