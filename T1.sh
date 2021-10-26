#python evaluate.py --experiment OldModel --model forecast_n5 --forecast_mode velocity_forward
#python evaluate.py --experiment OldModel --model forecast_n5r --forecast_mode velocity_reverse
python evaluate.py --experiment OldModel --model forecast_n5 --extractBox --forecast_mode velocity_forward
python evaluate.py --experiment OldModel --model forecast_n5r --extractBox --forecast_mode velocity_reverse