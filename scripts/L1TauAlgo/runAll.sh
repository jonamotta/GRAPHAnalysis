# python scripts/L1TauAlgo_new/0_matchGen2cl3d.py --FE threshold --doTrainValid --testRun
# echo " "
# echo " "

python scripts/L1TauAlgo_new/1_calibrateCl3d.py --FE threshold --doPlots 
echo " "
echo " "

python scripts/L1TauAlgo_new/2_PUrejectionBDT.py --FE threshold --doPlots --doEfficiency #--doRescale
echo " "
echo " "

# python scripts/L1TauAlgo_new/3_ISOcalculation.py --FE threshold --hardPUrej 99 #--doRescale
# python scripts/L1TauAlgo_new/3_ISOcalculation.py --FE threshold --hardPUrej 95 #--doRescale
python scripts/L1TauAlgo_new/3_ISOcalculation.py --FE threshold --hardPUrej 90 #--doRescale
echo " "
echo " "

python scripts/L1TauAlgo_new/4_isoQCDrejection.py --FE threshold --PUWP 99 --doPlots --doEfficiency --hardPUrej 99 #--doRescale
python scripts/L1TauAlgo_new/4_isoQCDrejection.py --FE threshold --PUWP 95 --doPlots --doEfficiency --hardPUrej 95 #--doRescale
python scripts/L1TauAlgo_new/4_isoQCDrejection.py --FE threshold --PUWP 90 --doPlots --doEfficiency --hardPUrej 90 #--doRescale
echo " "
echo " "

# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 10 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 15 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 20 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 90 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 95 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 99 --doPlots --doRescale
# echo " "
# echo " "

# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 10 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 15 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 20 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 90 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 95 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 99 --doPlots --doRescale
# echo " "
# echo " "

# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 10 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 15 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 20 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 90 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 95 --doPlots --doRescale
# python scripts/L1TauAlgo_new/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 99 --doPlots --doRescale
# echo " "
# echo " "

# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 10 --hardPUrej 99 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 15 --hardPUrej 99 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 20 --hardPUrej 99 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 90 --hardPUrej 99 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 95 --hardPUrej 99 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 99 --hardPUrej 99 #--doRescale
# echo " "
# echo " "

# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 10 --hardPUrej 95 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 15 --hardPUrej 95 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 20 --hardPUrej 95 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 90 --hardPUrej 95 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 95 --hardPUrej 95 #--doRescale
# python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 99 --hardPUrej 95 #--doRescale
# echo " "
# echo " "

python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 10 --hardPUrej 90 #--doRescale
python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 15 --hardPUrej 90 #--doRescale
python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 20 --hardPUrej 90 #--doRescale
python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 90 --hardPUrej 90 #--doRescale
python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 95 --hardPUrej 90 #--doRescale
python scripts/L1TauAlgo_new/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 99 --hardPUrej 90 #--doRescale