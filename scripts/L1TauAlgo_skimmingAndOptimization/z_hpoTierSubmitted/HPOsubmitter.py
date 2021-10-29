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
    parser.add_option('--BDT', dest='BDT', help='BDT to optimize (PU, ISO)', default='PU')
    parser.add_option('--FE', dest='FE', help='front-end', default='threshold')
    parser.add_option('--min_trees', dest='min_trees', help='minimum number of boosting rounds', default=None)
    parser.add_option('--max_trees', dest='max_trees', help='maximum number of boosting rounds', default=None)
    parser.add_option('--init_points', dest='init_points',  help='number of initialization points for the bayesian optimization', default=5)
    parser.add_option('--n_iter', dest='n_iter',  help='na,ber of bayesian optimization rounds', default=50)
    parser.add_option('--tag', dest='tag', help='output folder tag name', default='')
    parser.add_option('--queue', dest='queue', help='batch queue', default='short')
    parser.add_option('--force', dest='force', help='replace existing optimization', action='store_true', default=False)
    parser.add_option('--sleep', dest='sleep', help='sleep in submission', default=False)
    (opt, args) = parser.parse_args()

    currFolder = os.getcwd ()

    if opt.BDT == 'PU':
        optimizer = 'scripts/hpoUtils/PUBDToptimizer.py'

    elif opt.BDT == 'ISO':
        optimizer = 'scripts/hpoUtils/ISOBDToptimizer.py'

    if opt.input[-1] == '/': opt.input = opt.input[:-1]
    if opt.output == 'none': opt.output = opt.input + '_HPO'

    if not os.path.exists(opt.input):
        print 'input folder', opt.input, 'not existing, exiting'
        sys.exit(1)
    if not opt.force and os.path.exists(opt.output):
        print 'output folder', opt.output, 'existing, exiting'
        sys.exit(1)
    elif os.path.exists(opt.output):
        os.system ('rm -rf ' + opt.output)
    os.system ('mkdir -p ' + opt.output)

    tagname = "/" + opt.tag if opt.tag else ''
    jobsDir = currFolder + tagname + '/HPO_' + basename(opt.input)
    if os.path.exists(jobsDir): os.system('rm -f ' + jobsDir + '/*')
    else: os.system('mkdir -p ' + jobsDir)

    commandFile = open(jobsDir + '/submit.sh', 'w')
    for num_trees in range(int(opt.min_trees),int(opt.max_trees)+1): 
        scriptFile = open('%s/optimizationJob_%d.sh'% (jobsDir,num_trees), 'w')
        scriptFile.write('#!/bin/bash\n')
        scriptFile.write('export X509_USER_PROXY=~/.t3/proxy.cert\n')
        scriptFile.write('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
        scriptFile.write('cd /home/llr/cms/motta/HGCAL/CMSSW_11_1_0/src\n')
        #scriptFile.write ('export SCRAM_ARCH=slc6_amd64_gcc472\n')
        scriptFile.write('eval `scram r -sh`\n')
        scriptFile.write('cd %s\n'%currFolder)
        scriptFile.write('source scripts/setup.sh\n')

        command = 'python ' + optimizer + ' --indir '+ opt.input + ' --FE ' + opt.FE + ' --num_trees ' + str(num_trees) + ' --init_points ' + opt.init_points + ' --n_iter ' + opt.n_iter
        command += ' >& ' + opt.output + '/' + "output_" + str(num_trees) + '.log\n'
        scriptFile.write(command)

        scriptFile.write('touch ' + jobsDir + '/done_%d\n'%num_trees)
        scriptFile.write('echo "All done for job %d" \n'%num_trees)
        scriptFile.close()
        os.system('chmod u+rwx %s/optimizationJob_%d.sh'% (jobsDir,num_trees))

        command = '/home/llr/cms/motta/t3submit -' + opt.queue + ' ' + jobsDir + '/optimizationJob_' + str(num_trees) + '.sh'
        if opt.sleep : time.sleep (0.1)
        os.system(command)
        commandFile.write(command + '\n')

    commandFile.close()