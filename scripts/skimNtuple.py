#!/usr/bin/env python

import os,sys
import optparse
import fileinput
import commands
import time
import glob
import subprocess
from os.path import basename
import ROOT

def parseInputFileList (fileName) :
    filelist = []
    with open (fileName) as fIn:
        for line in fIn:
            line = (line.split("#")[0]).strip()
            if line:
                filelist.append(line)
    return filelist

if __name__ == "__main__":
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--input', dest='input', help='input folder', default='none')
    parser.add_option('--output', dest='output', help='output folder', default='none')
    parser.add_option('--tag', dest='tag', help='output folder tag name', default='')
    parser.add_option('--queue', dest='queue', help='batch queue', default='short')
    parser.add_option('--njobs', dest='njobs', help='number of skim jobs', default=100, type = int)
    parser.add_option('--force', dest='force', help='replace existing reduced ntuples', action='store_true', default=False)
    parser.add_option('--sleep', dest='sleep', help='sleep in submission', default=False)
    parser.add_option('--isTau', dest='isTau', help='is the skimming done for tau leptons from HH2b2tau datasets?', default=0)
    parser.add_option('--isQCD', dest='isQCD', help='is the skimming done for tau leptons from MultiTau datasets?', default=0)
    parser.add_option('--nEvents', dest='nEvents', help='number of events to process', default=-1)
    parser.add_option('--DEBUG', dest='DEBUG', help='print all DEBUG information from skimming process', default=0)
    (opt, args) = parser.parse_args()

    currFolder = os.getcwd ()

    skimmer = 'skimNtuple.exe'

    if opt.input[-1] == '/': opt.input = opt.input[:-1]
    if opt.output == 'none': opt.output = opt.input + '_SKIM'

    if not os.path.exists(opt.input):
        print 'input folder', opt.input, 'not existing, exiting'
        sys.exit (1)
    if not opt.force and os.path.exists(opt.output):
        print 'output folder', opt.output, 'existing, exiting'
        sys.exit (1)
    elif os.path.exists(opt.output):
        os.system ('rm -rf ' + opt.output)
    os.system ('mkdir ' + opt.output)

    inputfiles = parseInputFileList(opt.input)
    if opt.njobs > len(inputfiles): opt.njobs = len(inputfiles)
    nfiles = (len(inputfiles) + len(inputfiles) % opt.njobs) / opt.njobs
    inputlists = [inputfiles[x:x+nfiles] for x in xrange (0, len(inputfiles), nfiles)]

    tagname = "/" + opt.tag if opt.tag else ''
    jobsDir = currFolder + tagname + '/SKIM_' + basename(opt.input)
    jobsDir = jobsDir.rstrip(".txt")
    if os.path.exists(jobsDir): os.system('rm -f ' + jobsDir + '/*')
    else: os.system('mkdir -p ' + jobsDir)

    n = int(0)
    commandFile = open(jobsDir + '/submit.sh', 'w')
    for listname in inputlists: 
        #create a wrapper for standalone cmssw job
        listFileName = "filelist_%i.txt" % n
        thisinputlistFile = open(jobsDir + "/" + listFileName, 'w')
        for line in listname:
            thisinputlistFile.write(line+"\n")
        thisinputlistFile.close()
        scriptFile = open('%s/skimJob_%d.sh'% (jobsDir,n), 'w')
        scriptFile.write('#!/bin/bash\n')
        scriptFile.write('export X509_USER_PROXY=~/.t3/proxy.cert\n')
        scriptFile.write('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
        scriptFile.write('cd /home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src\n')
        #scriptFile.write ('export SCRAM_ARCH=slc6_amd64_gcc472\n')
        scriptFile.write('eval `scram r -sh`\n')
        scriptFile.write('cd %s\n'%currFolder)
        scriptFile.write('source scripts/setup.sh\n')

        command = skimmer + ' ' + jobsDir+"/"+listFileName + ' ' + opt.output + '/' + "output_"+str(n)+".root" + ' ' + str(opt.nEvents) + ' ' + str(opt.isTau) + ' ' + str(opt.isQCD) + ' ' + str(opt.DEBUG)
        command += ' >& ' + opt.output + '/' + "output_" + str(n) + '.log\n'
        scriptFile.write(command)

        scriptFile.write('touch ' + jobsDir + '/done_%d\n'%n)
        scriptFile.write('echo "All done for job %d" \n'%n)
        scriptFile.close()
        os.system('chmod u+rwx %s/skimJob_%d.sh'% (jobsDir,n))

        command = '/home/llr/cms/motta/t3submit -' + opt.queue + ' ' + jobsDir + '/skimJob_' + str(n) + '.sh'
        if opt.sleep : time.sleep (0.1)
        os.system(command)
        commandFile.write(command + '\n')
        n = n + 1

    commandFile.close ()