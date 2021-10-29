# python scripts/L1TauAlgo_newNoCal/1_matchGen2cl3d.py --FE threshold --doTrainValid --testRun
# echo " "
# echo " "

python scripts/L1TauAlgo_newNoCal/2_PUrejectionBDT.py --FE threshold --doPlots --doEfficiency #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/3_ISOcalculation.py --FE threshold #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/4_isoQCDrejection.py --FE threshold --PUWP 99 --doPlots --doEfficiency #--doRescale
python scripts/L1TauAlgo_newNoCal/4_isoQCDrejection.py --FE threshold --PUWP 95 --doPlots --doEfficiency #--doRescale
python scripts/L1TauAlgo_newNoCal/4_isoQCDrejection.py --FE threshold --PUWP 90 --doPlots --doEfficiency #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 01 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 05 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 10 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 15 --doPlots #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 01 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 05 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 10 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 15 --doPlots #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 01 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 05 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 10 --doPlots #--doRescale
python scripts/L1TauAlgo_newNoCal/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 15 --doPlots #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 01 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 05 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 10 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 15 #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 01 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 05 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 10 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 15 #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 01 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 05 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 10 #--doRescale
python scripts/L1TauAlgo_newNoCal/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 15 #--doRescale