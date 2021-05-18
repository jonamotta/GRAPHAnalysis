import ROOT
from ROOT import *
import argparse
import os

# parse user's options
parser = argparse.ArgumentParser(description='Command line parser of plotting options')
parser.add_argument('--skimIter', dest='iter', help='directory conaining SKIM files', default=None)
parser.add_argument('--skimDir', dest='dir', help='directory conaining SKIM files', default=None)
# store parsed options
args = parser.parse_args()

if not args.iter:
    print('** WARNING: no iteration provided - EXITING')
    exit()
if not args.dir:
    print('** WARNING: no directory provided - EXITING')
    exit()

skim_base = '/data_CMS_upgrade/motta/HGCAL_SKIMS'

if not os.path.isdir(skim_base+'/'+args.iter+'/'+args.dir):
    print('** WARNING: directory ' + skim_base+'/'+args.iter+'/'+args.dir + ' does not exist - EXITING')
    exit()
if not os.path.isfile(skim_base+'/'+args.iter+'/'+args.dir+'/goodfiles.txt'):
    print('** WARNING: file ' + skim_base+'/'+args.iter+'/'+args.dir+'/goodfiles.txt' + ' does not exist - EXITING')
    exit()

goodfiles = open(skim_base+'/'+args.iter+'/'+args.dir+'/goodfiles.txt')

chain = TChain("HGCALskimmedTree")

for file in goodfiles.readlines():
    chain.Add(file.strip())

outFile = TFile.Open(skim_base+'/'+args.iter+'/'+args.dir+'/'+'mergedOutput.root',"RECREATE")
print('** INFO: merging from ' + skim_base+'/'+args.iter+'/'+args.dir+'/goodfiles.txt' + ' into ' + skim_base+'/'+args.iter+'/'+args.dir+'/'+'mergedOutput.root')
chain.Merge(outFile, -1)