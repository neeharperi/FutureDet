#Table 1
#python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --static_only --tp_pct 0.6 --cohort_analysis
#python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --static_only --tp_pct 0.6 --eval_only 
#python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --static_only --tp_pct 0.9 --eval_only 
#python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --static_only --tp_pct -1 --eval_only 

#python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward  --tp_pct 0.6 --eval_only
#python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward   --tp_pct 0.9 --eval_only 
#python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward   --tp_pct -1 --eval_only 

#Table 2
#python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --cohort_analysis
#python evaluate.py --experiment FutureDetection --model forecast_n0 --forecast_mode velocity_constant  --K 5 --cohort_analysis --eval_only

#python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --cohort_analysis 
#python evaluate.py --experiment FutureDetection --model forecast_n3 --forecast_mode velocity_forward --K 5 --cohort_analysis --eval_only 
####
#python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis
#python evaluate.py --experiment FutureDetection --model forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --K 5 --eval_only 

#python evaluate.py --experiment FutureDetection --model pp_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis
#python evaluate.py --experiment FutureDetection --model pp_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --K 5 --eval_only 

#python evaluate.py --experiment FutureDetection --model forecast_n3dtfm --forecast_mode velocity_dense --cohort_analysis
#python evaluate.py --experiment FutureDetection --model forecast_n3dtfm --forecast_mode velocity_dense --cohort_analysis --K 5 --eval_only 

#Table 3
#python evaluate.py --experiment FutureDetection --model forecast_n3d --forecast_mode velocity_dense --cohort_analysis 
#python evaluate.py --experiment FutureDetection --model forecast_n3d --forecast_mode velocity_dense --K 5 --cohort_analysis --eval_only 

#python evaluate.py --experiment FutureDetection --model forecast_n3dt --forecast_mode velocity_dense --cohort_analysis 
#python evaluate.py --experiment FutureDetection --model forecast_n3dt --forecast_mode velocity_dense --K 5 --cohort_analysis --eval_only 

#Table 4
#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant  --cohort_analysis --classname pedestrian
#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n0 --forecast_mode velocity_constant  --K 5 --cohort_analysis --eval_only --classname pedestrian

#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward --cohort_analysis --classname pedestrian
#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3 --forecast_mode velocity_forward --K 5 --cohort_analysis --eval_only --classname pedestrian

#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --classname pedestrian
#python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --K 5 --eval_only --classname pedestrian

#python evaluate.py --experiment FutureDetection --model pp_pedestrian_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --classname pedestrian
#python evaluate.py --experiment FutureDetection --model pp_pedestrian_forecast_n3dtf --forecast_mode velocity_dense --cohort_analysis --K 5 --eval_only --classname pedestrian

python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtfm --forecast_mode velocity_dense --cohort_analysis --classname pedestrian
python evaluate.py --experiment FutureDetection --model pedestrian_forecast_n3dtfm --forecast_mode velocity_dense --cohort_analysis --K 5 --eval_only --classname pedestrian
