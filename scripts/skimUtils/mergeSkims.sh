#SKIMDIRS="SKIM_GluGluHHTo2b2Tau_PU200_4iso SKIM_QCD_PU200_4iso SKIM_RelValNu_PU200 SKIM_RelValQCD_noPU_4iso SKIM_RelValQCD_PU200_4iso SKIM_RelValSingleTau_noPU_4iso SKIM_RelValSingleTau_noPU_caloTruth SKIM_RelValSingleTau_PU200_4iso SKIM_RelValSingleTau_PU200_caloTruth SKIM_RelValTenTau_noPU_4iso SKIM_RelValTenTau_noPU_caloTruth SKIM_RelValTenTau_PU200_4iso SKIM_RelValTenTau_PU200_caloTruth"
SKIMDIRS="SKIM_RelValQCD_noPU_4iso SKIM_RelValQCD_PU200_4iso SKIM_QCD_PU200_4iso SKIM_RelValSingleTau_PU200_4iso"
SKIMITER="SKIM_12May2021"

for dir in ${SKIMDIRS}
do
    python scripts/skimUtils/mergeSkims.py --skimIter ${SKIMITER} --skimDir ${dir}
    sleep 5
done

