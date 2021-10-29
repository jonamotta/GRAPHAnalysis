python scripts/L1TauAlgo/2_calibrateCl3d.py --FE threshold --doPlots --ptcut 10
echo " "
echo " "

python scripts/L1TauAlgo/3_PUrejectionBDT.py --FE threshold --doPlots --doEfficiency
echo " "
echo " "

python scripts/L1TauAlgo/4_ISOcalculation.py --FE threshold
echo " "
echo " "

python scripts/L1TauAlgo/5_isoQCDrejection.py --FE threshold --PUWP 99 --doPlots --doEfficiency
python scripts/L1TauAlgo/5_isoQCDrejection.py --FE threshold --PUWP 95 --doPlots --doEfficiency
python scripts/L1TauAlgo/5_isoQCDrejection.py --FE threshold --PUWP 90 --doPlots --doEfficiency
echo " "
echo " "

python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 01 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 05 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 10 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 15 --doPlots
echo " "
echo " "

python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 01 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 05 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 10 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 15 --doPlots
echo " "
echo " "

python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 01 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 05 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 10 --doPlots
python scripts/L1TauAlgo/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 15 --doPlots
echo " "
echo " "

python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 01
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 05
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 10
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 15
echo " "
echo " "

python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 01
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 05
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 10
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 15
echo " "
echo " "

python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 01
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 05
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 10
python scripts/L1TauAlgo/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 15