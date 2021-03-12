#SKIMDIRS="SKIM_GluGluHHTo2b2Tau_PU200 SKIM_RelValNu_PU200 SKIM_RelValTenTau_PU200"
SKIMDIRS="SKIM_QCD_PU200"
SKIMITER="SKIM_18Feb2021"

for dir in ${SKIMDIRS}
do
    python scripts/mergeSkims.py --skimIter ${SKIMITER} --skimDir ${dir}
    sleep 5
done
