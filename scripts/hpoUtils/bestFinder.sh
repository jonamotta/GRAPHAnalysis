OPTDIR="/data_CMS_upgrade/motta/HGCAL/hpoBDTs"
OUTDIR="PUrejectionBDT"
INDIR="/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis"

TYPE="L1BDT_redCalib"
TAG="HPO_13Sep2021"

cd $OPTDIR/$OUTDIR/$TYPE/$TAG/calibrated_C3

if [[ -f best_params.txt ]]
then
    rm best_params.txt
fi

best=0
touch best_params.txt

for logfile in output_*.log
do
    if [[ ! -s $logfile ]]; then
        echo "$logfile empty --> need relaunch"
        continue
    fi
    
    tmp=$(tail -n 1 $logfile)
    
    if (( ${best##*.} < ${tmp##*.} ))
    then
        best=$tmp
        params=$(tail -n 2 $logfile)

        f=${logfile#*_}
        idx=${f%.*}
    fi
done

echo "BOOSTING ROUNDS" >> best_params.txt
echo $idx >> best_params.txt
echo "PARAMETERS" >> best_params.txt
echo $params >> best_params.txt
cd -