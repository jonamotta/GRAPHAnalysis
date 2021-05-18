python scripts/L1TauAlgo/1_matchGen2cl3d.py --FE threshold --doTrainValid
sleep 5

python scripts/L1TauAlgo/2_calibrateCl3d.py --FE threshold --doPlots
sleep 5

python scripts/L1TauAlgo/3_PUrejectionBDT.py --FE threshold --doPlots
sleep 5

python scripts/L1TauAlgo/4A2_DMsortingBDT.py --FE threshold --doPlots --WP 99
sleep 5

python scripts/L1TauAlgo/4A2_DMsortingBDT.py --FE threshold --doPlots --WP 95
sleep 5

python scripts/L1TauAlgo/4A2_DMsortingBDT.py --FE threshold --doPlots --WP 90
sleep 5