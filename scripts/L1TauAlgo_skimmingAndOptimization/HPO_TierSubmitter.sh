source scripts/setup.sh
source /opt/exp_soft/cms/t3/t3setup

INDIR="/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis"
OPTDIR="/data_CMS_upgrade/motta/HGCAL/hpoBDTs"
OUTDIR="PUrejectionBDT"
TAG="HPO_2021_11_06"
FE="threshold"

mkdir -p $OPTDIR/$OUTDIR
python scripts/L1TauAlgo_skimmingAndOptimization/HPO_TierSubmitter.py --force --input $INDIR --output $OPTDIR/$OUTDIR/$TAG/$FE --BDT PU --FE $FE --min_trees 5 --max_trees 100 --doRescale --tag $TAG --queue long --sleep True



