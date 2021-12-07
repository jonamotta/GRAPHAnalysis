import os, sys
from subprocess import Popen, PIPE
import datetime

# ====================================================================


tag1 = "/GluGluToHHTo2B2Tau_v11_1_PU200"
tag2 = "/RelValTenTau_v11_1_PU200"
tag3 = "/RelValSingleTau_v11_1_PU200"
tag4 = "/RelValNu_v11_1_PU200"
tag5 = "/Minbias_v11_1_PU200"
tag6 = "/RelValQCD_v11_1_PU200"
tag7 = "/VBFHToTauTau_v11_1_PU200"
tag8 = "/VBFHToInv_v11_1_PU200"
tag9 = "/ZprimeToTauTau_v11_1_PU200"

tags = [tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9]

# ====================================================================

outFolder1 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/GluGluToHHTo2B2Tau_v11_1'
outFolder2 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValTenTau_v11_1'
outFolder3 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValSingleTau_v11_1'
outFolder4 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValNu_v11_1'
outFolder5 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/Minbias_v11_1'
outFolder6 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/RelValQCD_v11_1'
outFolder7 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/VBFHToTauTau_v11_1'
outFolder8 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/VBFHToInv_v11_1'
outFolder9 = '/home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src/GRAPHAnalysis/inputFiles/ZprimeToTauTau_v11_1'

outFolders = [outFolder1, outFolder2, outFolder3, outFolder4, outFolder5, outFolder6, outFolder7, outFolder8, outFolder9]

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

useOnly = [] #empty list to do list for all the folders

dpmHome = '/dpm/in2p3.fr/home/cms/trivcat'
partialPath1 = "/store/user/jmotta/HGCAL/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5"
partialPath2 = '/store/user/jmotta/HGCAL/RelValTenTau_15_500'
partialPath3 = '/store/user/jmotta/HGCAL/RelValSingleTauFlatPt2To150'
partialPath4 = "/store/user/jmotta/HGCAL/RelValNuGun"
partialPath5 = "/store/user/jmotta/HGCAL/MinBias_TuneCP5_14TeV-pythia8"
partialPath6 = "/store/user/jmotta/HGCAL/RelValQCD_FlatPt_15_3000HS_14"
partialPath7 = "/store/user/jmotta/HGCAL/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8"
partialPath8 = "/store/user/jmotta/HGCAL/VBF_HToInvisible_M125_14TeV_powheg_pythia8_TuneCP5"
partialPath9 = "/store/user/jmotta/HGCAL/ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola"

partialPaths = [partialPath1, partialPath2, partialPath3, partialPath4, partialPath5, partialPath6, partialPath7, partialPath8, partialPath9]

for i in range(len(partialPaths))
    partialPath = partialPaths[i]
    tag = tags[i]
    outFolder = outFolders[i]
    os.system('mkdir -p '+outFolder)
    
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
