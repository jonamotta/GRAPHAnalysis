#python scripts/L1TauAlgo_BDTbasedISO/2_calibrateCl3d.py --FE threshold --doPlots
#echo " "
#echo " "
#
#python scripts/L1TauAlgo_BDTbasedISO/3_PUrejectionBDT.py --FE threshold --doPlots
#echo " "
#echo " "
#
#python scripts/L1TauAlgo_BDTbasedISO/4_ISOcalculation.py --FE threshold
#echo " "
#echo " "
#
#python scripts/L1TauAlgo_BDTbasedISO/5_isoQCDrejection.py --FE threshold --PUWP 99 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/5_isoQCDrejection.py --FE threshold --PUWP 95 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/5_isoQCDrejection.py --FE threshold --PUWP 90 --doPlots
#echo " "
#echo " "
#
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 01 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 05 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 99 --ISOWP 10 --doPlots
#echo " "
#echo " "
#
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 01 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 05 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 95 --ISOWP 10 --doPlots
#echo " "
#echo " "
#
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 01 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 05 --doPlots
#python scripts/L1TauAlgo_BDTbasedISO/6_DMsortingBDT.py --FE threshold --PUWP 90 --ISOWP 10 --doPlots
#echo " "
#echo " "

#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 01
#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 05
#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 99 --ISOWP 10

#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 01
#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 05
#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 95 --ISOWP 10

#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 01
#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 05
#python scripts/L1TauAlgo_BDTbasedISO/7_turnONs.py --FE threshold --PUWP 90 --ISOWP 10

python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 90 --ISOWP 01
python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 90 --ISOWP 05
python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 90 --ISOWP 10

python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 95 --ISOWP 01
python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 95 --ISOWP 05
python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 95 --ISOWP 10

python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 99 --ISOWP 01
python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 99 --ISOWP 05
python scripts/L1TauIso_test/8_rates.py --FE threshold --PUWP 99 --ISOWP 10