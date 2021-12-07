# python scripts/L1TauAlgo_reducedEtaCoverage/0_matchGen2cl3d.py --FE threshold --doTrainValid
# echo " "
# echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/1_calibrateCl3d.py --FE threshold --doPlots 
echo " "
echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/2_PUrejectionBDT.py --FE threshold --doPlots --doEfficiency --doRescale
echo " "
echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/3_ISOcalculation.py --FE threshold --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/3_ISOcalculation.py --FE threshold --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/3_ISOcalculation.py --FE threshold --doRescale --hardPUrej 90
echo " "
echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/4_isoQCDrejection.py --FE threshold --PUWP 99 --doPlots --doEfficiency --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/4_isoQCDrejection.py --FE threshold --PUWP 95 --doPlots --doEfficiency --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/4_isoQCDrejection.py --FE threshold --PUWP 90 --doPlots --doEfficiency --doRescale --hardPUrej 90
echo " "
echo " "

# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 10 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 15 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 20 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 90 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 95 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 99 --doPlots --doRescale
# echo " "
# echo " "

# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 10 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 15 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 20 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 90 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 95 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 99 --doPlots --doRescale
# echo " "
# echo " "

# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 10 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 15 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 20 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 90 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 95 --doPlots --doRescale
# python scripts/L1TauAlgo_reducedEtaCoverage/5_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 99 --doPlots --doRescale
# echo " "
# echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 10 --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 15 --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 20 --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 90 --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 95 --doRescale --hardPUrej 99
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 99 --ISOWP 99 --doRescale --hardPUrej 99
echo " "
echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 10 --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 15 --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 20 --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 90 --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 95 --doRescale --hardPUrej 95
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 95 --ISOWP 99 --doRescale --hardPUrej 95
echo " "
echo " "

python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 10 --doRescale --hardPUrej 90
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 15 --doRescale --hardPUrej 90
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 20 --doRescale --hardPUrej 90
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 90 --doRescale --hardPUrej 90
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 95 --doRescale --hardPUrej 90
python scripts/L1TauAlgo_reducedEtaCoverage/6_turnONs.py --FE threshold --PUWP 90 --ISOWP 99 --doRescale --hardPUrej 90