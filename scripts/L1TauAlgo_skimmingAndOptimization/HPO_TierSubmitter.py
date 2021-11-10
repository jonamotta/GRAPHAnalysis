#!/usr/bin/env python

import os,sys
import optparse
import time
from os.path import basename

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
    parser.add_option('--input', dest='input', help='inut folder', default='none')
    parser.add_option('--output', dest='output', help='output folder', default='none')
    parser.add_option('--BDT', dest='BDT', help='BDT to optimize (PU, ISO)', default='PU')
    parser.add_option('--FE', dest='FE', help='front-end', default='threshold')
    parser.add_option('--min_trees', dest='min_trees', help='minimum number of boosting rounds', default=None)
    parser.add_option('--max_trees', dest='max_trees', help='maximum number of boosting rounds', default=None)
    parser.add_option('--doRescale', dest='doRescale', help='do you want rescale the features?', action='store_true', default=False)
    parser.add_option('--tag', dest='tag', help='output folder tag name', default='')
    parser.add_option('--queue', dest='queue', help='batch queue', default='short')
    parser.add_option('--force', dest='force', help='replace existing optimization', action='store_true', default=False)
    parser.add_option('--sleep', dest='sleep', help='sleep in submission', default=False)
    (opt, args) = parser.parse_args()

    currFolder = os.getcwd ()

    if opt.BDT == 'PU':
        optimizer = 'scripts/L1TauAlgo_skimmingAndOptimization/HPO4Tier_PUBDT.py'

    elif opt.BDT == 'ISO':
        optimizer = 'scripts/L1TauAlgo_skimmingAndOptimization/HPO4Tier_QCDBDT.py'

    if opt.input[-1] == '/': opt.input = opt.input[:-1]
    if opt.output == 'none': opt.output = opt.input + '_HPO'

    if not os.path.exists(opt.input):
        print('input folder', opt.input, 'not existing, exiting')
        sys.exit(1)
    if not opt.force and os.path.exists(opt.output):
        print('output folder', opt.output, 'existing, exiting')
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
        scriptFile.write('eval `scram r -sh`\n')
        scriptFile.write('cd %s\n'%currFolder)
        scriptFile.write('source scripts/setup.sh\n')
        #scriptFile.write('module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7')
        #scriptFile.write('module load python/3.7.0')
        #scriptFile.write('source /opt/exp_soft/llr/root/v6.14.04-el7-gcc71-py37/bin/thisroot.sh')

        command = 'python ' + optimizer + ' --FE ' + opt.FE + ' --num_trees ' + str(num_trees) + ' --output ' + opt.output
        if opt.doRescale: command += ' --doRescale'
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