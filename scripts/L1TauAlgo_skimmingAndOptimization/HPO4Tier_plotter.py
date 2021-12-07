from matplotlib import pyplot as plt


files = ["HPO_PUBDT_BestParamsScores_br5.txt", "HPO_PUBDT_BestParamsScores_br6.txt", "HPO_PUBDT_BestParamsScores_br7.txt", "HPO_PUBDT_BestParamsScores_br8.txt", "HPO_PUBDT_BestParamsScores_br9.txt", "HPO_PUBDT_BestParamsScores_br10.txt", "HPO_PUBDT_BestParamsScores_br11.txt", "HPO_PUBDT_BestParamsScores_br12.txt", "HPO_PUBDT_BestParamsScores_br13.txt", "HPO_PUBDT_BestParamsScores_br14.txt", "HPO_PUBDT_BestParamsScores_br15.txt", "HPO_PUBDT_BestParamsScores_br16.txt", "HPO_PUBDT_BestParamsScores_br17.txt", "HPO_PUBDT_BestParamsScores_br18.txt", "HPO_PUBDT_BestParamsScores_br19.txt", "HPO_PUBDT_BestParamsScores_br20.txt", "HPO_PUBDT_BestParamsScores_br21.txt", "HPO_PUBDT_BestParamsScores_br22.txt", "HPO_PUBDT_BestParamsScores_br23.txt", "HPO_PUBDT_BestParamsScores_br24.txt", "HPO_PUBDT_BestParamsScores_br25.txt", "HPO_PUBDT_BestParamsScores_br26.txt", "HPO_PUBDT_BestParamsScores_br27.txt", "HPO_PUBDT_BestParamsScores_br28.txt", "HPO_PUBDT_BestParamsScores_br29.txt", "HPO_PUBDT_BestParamsScores_br30.txt", "HPO_PUBDT_BestParamsScores_br31.txt", "HPO_PUBDT_BestParamsScores_br32.txt", "HPO_PUBDT_BestParamsScores_br33.txt", "HPO_PUBDT_BestParamsScores_br34.txt", "HPO_PUBDT_BestParamsScores_br35.txt", "HPO_PUBDT_BestParamsScores_br36.txt", "HPO_PUBDT_BestParamsScores_br37.txt", "HPO_PUBDT_BestParamsScores_br38.txt", "HPO_PUBDT_BestParamsScores_br40.txt", "HPO_PUBDT_BestParamsScores_br41.txt", "HPO_PUBDT_BestParamsScores_br42.txt", "HPO_PUBDT_BestParamsScores_br43.txt", "HPO_PUBDT_BestParamsScores_br44.txt", "HPO_PUBDT_BestParamsScores_br45.txt", "HPO_PUBDT_BestParamsScores_br46.txt", "HPO_PUBDT_BestParamsScores_br47.txt", "HPO_PUBDT_BestParamsScores_br48.txt", "HPO_PUBDT_BestParamsScores_br49.txt", "HPO_PUBDT_BestParamsScores_br52.txt", "HPO_PUBDT_BestParamsScores_br53.txt", "HPO_PUBDT_BestParamsScores_br54.txt", "HPO_PUBDT_BestParamsScores_br55.txt", "HPO_PUBDT_BestParamsScores_br56.txt", "HPO_PUBDT_BestParamsScores_br57.txt", "HPO_PUBDT_BestParamsScores_br58.txt", "HPO_PUBDT_BestParamsScores_br59.txt", "HPO_PUBDT_BestParamsScores_br63.txt", "HPO_PUBDT_BestParamsScores_br64.txt", "HPO_PUBDT_BestParamsScores_br65.txt", "HPO_PUBDT_BestParamsScores_br66.txt", "HPO_PUBDT_BestParamsScores_br67.txt", "HPO_PUBDT_BestParamsScores_br68.txt", "HPO_PUBDT_BestParamsScores_br69.txt", "HPO_PUBDT_BestParamsScores_br70.txt", "HPO_PUBDT_BestParamsScores_br74.txt", "HPO_PUBDT_BestParamsScores_br75.txt", "HPO_PUBDT_BestParamsScores_br76.txt", "HPO_PUBDT_BestParamsScores_br77.txt", "HPO_PUBDT_BestParamsScores_br78.txt", "HPO_PUBDT_BestParamsScores_br79.txt", "HPO_PUBDT_BestParamsScores_br80.txt", "HPO_PUBDT_BestParamsScores_br81.txt", "HPO_PUBDT_BestParamsScores_br85.txt", "HPO_PUBDT_BestParamsScores_br86.txt", "HPO_PUBDT_BestParamsScores_br87.txt", "HPO_PUBDT_BestParamsScores_br88.txt", "HPO_PUBDT_BestParamsScores_br89.txt", "HPO_PUBDT_BestParamsScores_br90.txt", "HPO_PUBDT_BestParamsScores_br91.txt", "HPO_PUBDT_BestParamsScores_br92.txt", "HPO_PUBDT_BestParamsScores_br96.txt", "HPO_PUBDT_BestParamsScores_br97.txt", "HPO_PUBDT_BestParamsScores_br98.txt", "HPO_PUBDT_BestParamsScores_br99.txt", "HPO_PUBDT_BestParamsScores_br100.txt"]

path = "/data_cms_upgrade/motta/HGCAL/hpoBDTs/PUrejectionBDT/HPO_2021_11_06/threshold/"

Xs = []
TRs = []
TEs = [] 

for file in files:
    x = file.split('.')[0].strip('HPO_PUBDT_BestParamsScores_br')
    Xs.append(int(x))

    with open(path+file) as f:
        lines = f.readlines()
        for l in lines:
            if "train-auc-mean" in l:
                TRs.append(float(l.split("=")[1].strip(" \n")))
            if "test-auc-mean" in l:
                TEs.append(float(l.split("=")[1].strip(" \n")))
                break

plt.figure(figsize=(10,10))
plt.plot(Xs, TRs, lw=2, marker="h", ms=5, label='train', color='blue')
plt.plot(Xs, TEs, lw=2, marker="h", ms=5, label='test', color='red')
plt.grid(linestyle=':')
plt.xlabel('Boosting rounds')
plt.ylabel(r'Optimization metric')
plt.savefig('test.pdf')
plt.close()