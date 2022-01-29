python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --cohort_analysis
python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --K 5 --cohort_analysis --eval_only 
python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --K 0 --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis 
python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --K 5 --cohort_analysis --eval_only 
python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --K 0 --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3t --forecast_mode velocity_forward --cohort_analysis 
python evaluate.py --experiment FutureDetection --model forecast_n3t --forecast_mode velocity_forward --K 5 --cohort_analysis --eval_only 
python evaluate.py --experiment FutureDetection --model forecast_n3t --forecast_mode velocity_forward --K 0 --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis
python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --K 5 --cohort_analysis --eval_only 
python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --K 0 --cohort_analysis --eval_only 

