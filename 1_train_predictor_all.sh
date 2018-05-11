
declare -a ecg_filenames=("chfdb_chf01_275.pkl"
			  "chfdb_chf13_45590.pkl"
			  "chfdbchf15.pkl"
			  "ltstdb_20221_43.pkl"
			  "ltstdb_20321_240.pkl"
			  "mitdb__100_180.pkl"
			  "qtdbsel102.pkl"
			  "stdb_308_0.pkl"
			  "xmitdb_x108_0.pkl"
			  )
declare -a respiration_filenames=("nprs43.pkl"
				  "nprs44.pkl"
       				  ) 
declare -a space_shuttle_filenames=("TEK14.pkl"
				    "TEK16.pkl"
       				    "TEK17.pkl") 
declare -a gesture_filenames=("ann_gun_CentroidA.pkl")
declare -a power_demand_filenames=("power_data.pkl")
declare -a nyc_taxi_filenames=("nyc_taxi.pkl")

for idx in "${ecg_filenames[@]}"
do
	ipython 1_train_predictor.py -- --data ecg --filename "$idx" --emsize 128 --nhid 128 --save_fig --resume --epoch 600
done

for idx in "${respiration_filenames[@]}"
do
	ipython 1_train_predictor.py -- --data respiration --filename "$idx" --emsize 128 --nhid 128 --save_fig --resume --epoch 600
done

for idx in "${space_shuttle_filenames[@]}"
do
	ipython 1_train_predictor.py -- --data space_shuttle --filename "$idx" --emsize 128 --nhid 128 --save_fig --resume --epoch 600
done

for idx in "${gesture_filenames[@]}"
do
	ipython 1_train_predictor.py -- --data gesture --filename "$idx" --emsize 128 --nhid 128 --save_fig --resume --epoch 600
done

for idx in "${power_demand_filenames[@]}"
do
	ipython 1_train_predictor.py -- --data power_demand --filename "$idx" --emsize 128 --nhid 128 --save_fig --resume --epoch 600
done

for idx in "${nyc_taxi_filenames[@]}"
do
	ipython 1_train_predictor.py -- --data nyc_taxi --filename "$idx" --emsize 128 --nhid 128 --save_fig --resume --epoch 600





