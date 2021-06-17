import os, sys
from subprocess import Popen, PIPE
import datetime

# ====================================================================

# tag = "/GluGluToHHTo2B2Tau_node_SM_PU200"
# tag = "/GluGluToHHTo2B2Tau_node_SM_PU200_caloTruth"
# tag = "/GluGluToHHTo2B2Tau_PU200_4iso"

# tag = "/RelValTenTau_PU200_caloTruth"
# tag = "/RelValTenTau_noPU_caloTruth"
# tag = "/RelValTenTau_PU200_4iso"
# tag = "/RelValTenTau_noPU_4iso"

# tag = "/RelValSingleTau_PU200_4iso"
# tag = "/RelValSingleTau_PU200_caloTruth"
# tag = "/RelValSingleTau_noPU_4iso"
# tag = "/RelValSingleTau_noPU_caloTruth"

# tag = "/RelValNu_PU200"
tag = "/Minibias_PU200"

# tag = "/QCD_PU200_4iso"
# tag = "/QCD_noPU_4iso"
# tag = "/RelValQCD_PU200_4iso"
# tag = "/RelValQCD_noPU_4iso"

# ====================================================================

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/GluGluToHHTo2B2Tau_v11'

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValTenTau_v11'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValSingleTau_v11'

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValNu_v11'

#outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/QCD_v11'

outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/Minbias_v11'

# ====================================================================

def formatName (name, pathToRem):
    name.strip(pathToRem)
    name = "root://polgrid4.in2p3.fr/" + name
    return name

def saveToFile (lista, filename):
    f = open (filename, "w")

    for elem in lista:
        f.write("%s\n" % elem) #vintage
    
    f.close()

# ====================================================================

os.system('mkdir -p '+outFolder)

useOnly = [] #empty list to do list fot all the folders

dpmHome = '/dpm/in2p3.fr/home/cms/trivcat'
# partialPath = "/store/user/jmotta/HGCAL/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5"
# partialPath = "/store/user/jmotta/HGCAL/RelValNuGun"
# partialPath = '/store/user/jmotta/HGCAL/RelValTenTau_15_500'
# partialPath = '/store/user/jmotta/HGCAL/RelValSingleTauFlatPt2To150'
# partialPath = "/store/user/jmotta/HGCAL/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8"
# partialPath = "/store/user/jmotta/HGCAL/RelValQCD_FlatPt_15_3000HS_14"
# partialPath = "/store/user/jmotta/HGCAL/RelValQCD_Pt15To7000_Flat_14"
partialPath = "/store/user/jmotta/HGCAL/MinBias_TuneCP5_14TeV-pythia8"

path = dpmHome + partialPath + tag
if outFolder[-1] != "/": outFolder += '/'

getDateTimeFolder = "/usr/bin/rfdir %s | awk '{print $9}'" % path
pipeDateTimeFolder = Popen(getDateTimeFolder, shell=True, stdout=PIPE)

allLists = {} #dictionary

for line in pipeDateTimeFolder.stdout:
    if useOnly:
        if not (line.strip()) in useOnly:
            continue

    samplesPath = (path + '/' + line.strip() + '/0000').strip() 
    sampleDateTime = line.strip()
    allLists[sampleDateTime] = []

    getFiles = "/usr/bin/rfdir %s | awk '{print $9}'" % samplesPath
    pipeFiles = Popen(getFiles, shell=True, stdout=PIPE)

    for file in pipeFiles.stdout:
        filePath = (samplesPath + '/' + file).strip()
        allLists[sampleDateTime].append(formatName(filePath,dpmHome))

for sample, lista in allLists.iteritems():
   if lista: 
      outName = outFolder + tag + '_' + sample.split('_')[0] + ".txt"
      saveToFile (lista, outName)
