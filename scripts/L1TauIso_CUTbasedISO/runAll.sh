python scripts/L1TauIso/skim2match.py --FE threshold --doNu
sleep 5

python scripts/L1TauIso/match2candidate.py --FE threshold --doNu
sleep 5

python scripts/L1TauIso/candidate2iso.py --FE threshold --doNu
sleep 5

python scripts/L1TauIso/iso2cut.py --FE threshold
sleep 5
