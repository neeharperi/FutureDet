python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant 
python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward 
python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3st --forecast_mode velocity_sparse_match
python evaluate.py --experiment FutureDetection --model forecast_n3st --forecast_mode velocity_sparse_match --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3st --forecast_mode velocity_sparse_match --eval_only --K 5
python evaluate.py --experiment FutureDetection --model forecast_n3st --forecast_mode velocity_sparse_match --cohort_analysis --eval_only -- 5

python evaluate.py --experiment FutureDetection --model forecast_n3dt --forecast_mode velocity_dense
python evaluate.py --experiment FutureDetection --model forecast_n3dt --forecast_mode velocity_dense --cohort_analysis --eval_only 

python evaluate.py --experiment FutureDetection --model forecast_n3dt --forecast_mode velocity_dense --eval_only --K 5
python evaluate.py --experiment FutureDetection --model forecast_n3dt --forecast_mode velocity_dense --cohort_analysis --eval_only --K 5 
