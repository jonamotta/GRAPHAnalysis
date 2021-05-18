AMESSAGE="V11 HGCAL SKIMS: \n - caloTruth CONTAIN CALO-TRUTH INFORMATION \n - ... DOES NOT CONTAIN CALO-TRUTH INFORMATION"

SKIMDIR="/data_CMS_upgrade/motta/HGCAL_SKIMS"
HASHDIR="/SKIM_12May2021"

source scripts/setup.sh
source /opt/exp_soft/cms/t3/t3setup
mkdir -p $SKIMDIR/$HASHDIR
touch $SKIMDIR/$HASHDIR/README.txt
echo -e $AMESSAGE > $SKIMDIR/$HASHDIR/README.txt
cp scripts/listAll.sh $SKIMDIR/$HASHDIR


###################################################################################################################################################################
###################################################################################################################################################################

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 1   --isQCD 0   --input inputFiles/GluGluToHHTo2B2Tau_v11/GluGluToHHTo2B2Tau_PU200_4iso_210510.txt   --output $SKIMDIR/$HASHDIR/SKIM_GluGluHHTo2b2Tau_PU200_4iso       --tag $HASHDIR/SKIM_GluGluHHTo2b2Tau_PU200_4iso      --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 1   --isTau 1   --isQCD 0   --input inputFiles/RelValTenTau_v11/RelValTenTau_PU200_caloTruth_210511.txt          --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_PU200_caloTruth      --tag $HASHDIR/SKIM_RelValTenTau_PU200_caloTruth     --njobs 20 --queue long --sleep True
#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 1   --isQCD 0   --input inputFiles/RelValTenTau_v11/RelValTenTau_PU200_4iso_210510.txt               --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_PU200_4iso           --tag $HASHDIR/SKIM_RelValTenTau_PU200_4iso          --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 1   --isTau 1   --isQCD 0   --input inputFiles/RelValTenTau_v11/RelValTenTau_noPU_caloTruth_210512.txt           --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_noPU_caloTruth       --tag $HASHDIR/SKIM_RelValTenTau_noPU_caloTruth      --njobs 20 --queue long --sleep True
#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 1   --isQCD 0   --input inputFiles/RelValTenTau_v11/RelValTenTau_noPU_4iso_210512.txt                --output $SKIMDIR/$HASHDIR/SKIM_RelValTenTau_noPU_4iso            --tag $HASHDIR/SKIM_RelValTenTau_noPU_4iso           --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 1   --isTau 1   --isQCD 0   --input inputFiles/RelValSingleTau_v11/RelValSingleTau_PU200_caloTruth_210510.txt    --output $SKIMDIR/$HASHDIR/SKIM_RelValSingleTau_PU200_caloTruth   --tag $HASHDIR/SKIM_RelValSingleTau_PU200_caloTruth  --njobs 20 --queue long --sleep True
#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 1   --isQCD 0   --input inputFiles/RelValSingleTau_v11/RelValSingleTau_PU200_4iso_210510.txt         --output $SKIMDIR/$HASHDIR/SKIM_RelValSingleTau_PU200_4iso        --tag $HASHDIR/SKIM_RelValSingleTau_PU200_4iso       --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 1   --isTau 1   --isQCD 0   --input inputFiles/RelValSingleTau_v11/RelValSingleTau_noPU_caloTruth_210510.txt     --output $SKIMDIR/$HASHDIR/SKIM_RelValSingleTau_noPU_caloTruth    --tag $HASHDIR/SKIM_RelValSingleTau_noPU_caloTruth   --njobs 20 --queue long --sleep True
#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 1   --isQCD 0   --input inputFiles/RelValSingleTau_v11/RelValSingleTau_noPU_4iso_210510.txt          --output $SKIMDIR/$HASHDIR/SKIM_RelValSingleTau_noPU_4iso         --tag $HASHDIR/SKIM_RelValSingleTau_noPU_4iso        --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 0   --isQCD 1   --input inputFiles/QCD_v11/QCD_PU200_4iso_210510.txt                                 --output $SKIMDIR/$HASHDIR/QCD_PU200_4iso                         --tag $HASHDIR/SKIM_QCD_PU200_4iso                   --njobs 20 --queue long --sleep True
python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 0   --isQCD 1   --input inputFiles/QCD_v11/QCD_noPU_4iso_210511.txt                                  --output $SKIMDIR/$HASHDIR/QCD_PU200_4iso                         --tag $HASHDIR/SKIM_QCD_PU200_4iso                   --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 0   --isQCD 1   --input inputFiles/QCD_v11/RelValQCD_PU200_4iso_210510.txt                           --output $SKIMDIR/$HASHDIR/RelValQCD_PU200_4iso                   --tag $HASHDIR/SKIM_RelValQCD_PU200_4iso             --njobs 20 --queue long --sleep True
#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 0   --isQCD 1   --input inputFiles/QCD_v11/RelValQCD_noPU_4iso_210511.txt                            --output $SKIMDIR/$HASHDIR/RelValQCD_noPU_4iso                    --tag $HASHDIR/SKIM_RelValQCD_noPU_4iso              --njobs 20 --queue long --sleep True

#python scripts/skimUtils/skimNtuple.py --gen3Dmatch 0   --isTau 0   --isQCD 0   --input inputFiles/RelValNu_v11/RelValNu_v11_PU200_caloTruth_210429.txt              --output $SKIMDIR/$HASHDIR/SKIM_RelValNu_PU200                    --tag $HASHDIR/SKIM_RelValNu_PU200                   --njobs 20 --queue long --sleep True
