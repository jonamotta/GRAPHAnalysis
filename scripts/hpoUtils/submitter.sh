source scripts/setup.sh
source /opt/exp_soft/cms/t3/t3setup

OPTDIR="/data_CMS_upgrade/motta/HGCAL/hpoBDTs"
OUTDIR="PUrejectionBDT"
INDIR="/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis"

TYPE="L1BDT_redCalib"
TAG="HPO_13Sep2021"

mkdir -p $OPTDIR/$OUTDIR
python scripts/hpoUtils/submitter.py --input $INDIR/$TYPE/hdf5dataframes/calibrated_C3/ --output $OPTDIR/$OUTDIR/$TYPE/$TAG/calibrated_C3 --BDT PU --FE threshold --min_trees 10 --max_trees 200 --init_points 10 --n_iter 100 --tag $TAG --queue long --sleep True
#python scripts/hpoUtils/submitter.py --force --input $INDIR/$TYPE/hdf5dataframes/calibrated_C3/ --output $OPTDIR/$OUTDIR/$TYPE/$TAG/calibrated_C3 --BDT PU --FE threshold --min_trees 2 --max_trees 4 --init_points 2 --n_iter 3 --tag $TAG --queue long --sleep True
