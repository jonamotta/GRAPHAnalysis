import os, sys
from subprocess import Popen, PIPE
import datetime

# ====================================================================

# tag = "/MultiTau_PT15to500_v11_noPU"
# tag = "/MultiTau_PT15to500_v11_PU140"
# tag = "/MultiTau_PT15to500_v11_PU200"

# tag = "/GluGluToHHTo2B2Tau_node_SM_PU200"
tag = "/GluGluToHHTo2B2Tau_node_SM_PU200_caloTruth"

# tag = "/RelValTenTau_PU200"
# tag = "/RelValTenTau_noPU"
# tag = "/RelValTenTau_PU200_caloTruth"
# tag = "/RelValTenTau_noPU_caloTruth"

# tag = "/RelValNu_PU200"
# tag = "/RelValNu_PU200_caloTruth"

# tag = "/QCD_PU200"
# tag = "/QCD_noPU"
# tag = "/QCD_PU200_caloTruth"
# tag = "/QCD_noPU_caloTruth"

# ====================================================================

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/MultiTau_PT15to500_v11_noPU'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/MultiTau_PT15to500_v11_PU140'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/MultiTau_PT15to500_v11_PU200'

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/GluGluToHHTo2B2Tau_v11_PU200'
outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/GluGluToHHTo2B2Tau_v11_PU200_caloTruth'

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValTenTau_v11_PU200'
# outFolder = "/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValTenTau_v11_noPU"
# outFolder = "/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValTenTau_v11_PU200_caloTruth"
# outFolder = "/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValTenTau_v11_noPU_caloTruth"

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValNu_v11_PU200'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValNu_v11_PU200_caloTruth'

# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/QCD_v11_PU200'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/QCD_v11_noPU'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/QCD_v11_PU200_caloTruth'
# outFolder = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/QCD_v11_noPU_caloTruth'

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
# partialPath = "/store/user/jmotta/HGCAL/MultiTau_PT15to500"
partialPath = "/store/user/jmotta/HGCAL/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5"
# partialPath = "/store/user/jmotta/HGCAL/RelValNuGun"
# partialPath = '/store/user/jmotta/HGCAL/RelValTenTau_15_500'
# partialPath = "/store/user/jmotta/HGCAL/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8"

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
      outName = outFolder + outFolder.split('inputFiles/')[1].strip('/') + '_' + sample.split('_')[0] + ".txt"
      saveToFile (lista, outName)
