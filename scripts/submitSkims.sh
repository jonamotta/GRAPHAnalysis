AMESSAGE="V11 HGCAL SKIMS WITH CALO-TRUTH INFORMATION"

SKIMDIR="/data_CMS_upgrade/motta/HGCAL_SKIMS"
HASHDIR="/SKIM_21Apr2021"

source scripts/setup.sh
source /opt/exp_soft/cms/t3/t3setup
mkdir -p $SKIMDIR/$HASHDIR
touch $SKIMDIR/$HASHDIR/README.txt
echo $AMESSAGE > $SKIMDIR/$HASHDIR/README.txt
cp scripts/listAll.sh $SKIMDIR/$HASHDIR


###################################################################################################################################################################
###################################################################################################################################################################

# THESE FOUR MIGHT BE COMPLETELY USELESS
# python scripts/skimNtuple.py --isTau 0 --isQCD 0 --input inputFiles/MultiTau_PT15to500_v11_noPU/MultiTau_PT15to500_v11_noPU_210129.txt --output $SKIMDIR/$HASHDIR/SKIM_MultiTau_noPU --tag $HASHDIR/SKIM_MultiTau --njobs 3 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 0 --isQCD 0 --input inputFiles/MultiTau_PT15to500_v11_PU140/MultiTau_PT15to500_v11_PU140_210202.txt --output $SKIMDIR/$HASHDIR/SKIM_MultiTau_PU140 --tag $HASHDIR/SKIM_MultiTau --njobs 60 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 0 --isQCD 0 --input inputFiles/MultiTau_PT15to500_v11_PU200/MultiTau_PT15to500_v11_PU200_210202.txt --output $SKIMDIR/$HASHDIR/SKIM_MultiTau_PU200 --tag $HASHDIR/SKIM_MultiTau --njobs 60 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 1 --isQCD 0 --input inputFiles/GluGluToHHTo2B2Tau_v11_PU140/ --output $SKIMDIR/$HASHDIR/SKIM_GluGluHHTo2b2Tau_PU140 --tag $HASHDIR/SKIM_GluGluHHTo2b2Tau --njobs 5 --queue long --sleep True


# THESE ARE THE ONES TO BE USED - NO CALO-TRUTH
# python scripts/skimNtuple.py --isTau 1 --isQCD 0 --input inputFiles/GluGluToHHTo2B2Tau_v11_PU200/GluGluToHHTo2B2Tau_v11_PU200_210208.txt --output $SKIMDIR/$HASHDIR/SKIM_GluGluHHTo2b2Tau_PU200 --tag $HASHDIR/SKIM_GluGluHHTo2b2Tau --njobs 5 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 1 --isQCD 0 --input inputFiles/RelValTenTau_v11_PU200/RelValTenTau_v11_PU200_210217.txt  --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_PU200 --tag $HASHDIR/SKIM_RelValTenTau --njobs 10 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 0 --isQCD 0 --input inputFiles/RelValNu_v11_PU200/RelValNu_v11_PU200_210217.txt --output $SKIMDIR/$HASHDIR/SKIM_RelValNu_PU200 --tag $HASHDIR/SKIM_RelValNu --njobs 20 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 0 --isQCD 1 --input inputFiles/QCD_v11_PU200/QCD_v11_PU200_210217.txt  --output $SKIMDIR/$HASHDIR/SKIM_QCD_PU200 --tag $HASHDIR/SKIM_QCD --njobs 20 --queue long --sleep True


# THESE ARE FOR THE VISUALIZATION OF EVENTS - NO CALO-TRUTH
# python scripts/skimNtuple.py --isTau 1 --isQCD 0 --input inputFiles/RelValTenTau_v11_noPU/RelValTenTau_v11_noPU_210331.txt  --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_noPU --tag $HASHDIR/SKIM_RelValTenTau --njobs 2 --queue long --sleep True
# python scripts/skimNtuple.py --isTau 0 --isQCD 1 --input inputFiles/QCD_v11_noPU/QCD_v11_noPU_210331.txt  --output $SKIMDIR/$HASHDIR/SKIM_QCD_noPU --tag $HASHDIR/SKIM_QCD --njobs 30 --queue long --sleep True


# THESE ARE THE ONES TO BE USED - WITH CALO-TRUTH
python scripts/skimNtuple.py --isTau 1 --isQCD 0 --input inputFiles/HHbbtautau/GluGluToHHTo2B2Tau_v11_PU200_caloTruth/GluGluToHHTo2B2Tau_v11_PU200_caloTruth_210419.txt --output $SKIMDIR/$HASHDIR/SKIM_GluGluHHTo2b2Tau_PU200_caloTruth --tag $HASHDIR/SKIM_GluGluHHTo2b2Tau_caloTruth --njobs 5 --queue long --sleep True

python scripts/skimNtuple.py --isTau 1 --isQCD 0 --input inputFiles/TauGun/RelValTenTau_v11_PU200_caloTruth/RelValTenTau_v11_PU200_caloTruth_210419.txt   --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_PU200_caloTruth     --tag $HASHDIR/SKIM_RelValTenTau_caloTruth     --njobs 10 --queue long --sleep True
python scripts/skimNtuple.py --force --isTau 1 --isQCD 0 --input inputFiles/TauGun/RelValTenTau_v11_noPU_caloTruth/RelValTenTau_v11_noPU_caloTruth_210419.txt     --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_noPU_caloTruth      --tag $HASHDIR/SKIM_RelValTenTau_caloTruth     --njobs 2 --queue long --sleep True

python scripts/skimNtuple.py --isTau 0 --isQCD 1 --input inputFiles/QCD/QCD_v11_PU200_caloTruth/QCD_v11_PU200_caloTruth_210418.txt                        --output $SKIMDIR/$HASHDIR/SKIM_QCD_PU200_caloTruth              --tag $HASHDIR/SKIM_QCD_caloTruth              --njobs 20 --queue long --sleep True
python scripts/skimNtuple.py --isTau 0 --isQCD 1 --input inputFiles/QCD/QCD_v11_noPU_caloTruth/QCD_v11_noPU_caloTruth_210418.txt                          --output $SKIMDIR/$HASHDIR/SKIM_QCD_noPU_caloTruth               --tag $HASHDIR/SKIM_QCD_caloTruth              --njobs 30 --queue long --sleep True

python scripts/skimNtuple.py --isTau 0 --isQCD 0 --input inputFiles/NuGun/RelValNu_v11_PU200_caloTruth/RelValNu_v11_PU200_caloTruth_210418.txt            --output $SKIMDIR/$HASHDIR/SKIM_RelValNu_PU200_caloTruth         --tag $HASHDIR/SKIM_RelValNu_caloTruth         --njobs 20 --queue long --sleep True
