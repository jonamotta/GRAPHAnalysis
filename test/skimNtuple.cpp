#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TString.h>
#include <TLorentzVector.h>

#include "bigTree.h"
#include "skimTree.h"
#include "skimUtils.h"

using namespace std;


int main (int argc, char** argv)
{
    if (argc < 5)
    {
      cerr << "missing input parameters : argc is: " << argc << endl ;
      cerr << "usage: " << argv[0]
           << " inputFileNameList outputFileName nEvents isTau isQCD gen3Dmatch DEBUG" << endl ;
      return 1;
    }

    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------
    // INITIAL SETUP FROM PARSE

    TString inputFile = argv[1] ;
    TString outputFile = argv[2] ;
    cout << "** INFO: inputFile  : " << inputFile << endl;
    cout << "** INFO: outputFile : " << outputFile << endl;
    
    int nEvents = atoi(argv[3]);
    int isTau = atoi(argv[4]);
    int isQCD = atoi(argv[5]);

    bool gen3Dmatch = false;
    string opt6 (argv[6]);
    if (opt6 == "1") gen3Dmatch = true;

    bool DEBUG = false;
    string opt7 (argv[6]);
    if (opt7 == "1") DEBUG = true;

    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------
    // SKIMMING

    // take input files
    TChain * bigChain = new TChain("hgcalTriggerNtuplizer/HGCalTriggerNtuple");
    appendFromFileList(bigChain, inputFile);
    bigChain->SetCacheSize(0);
    bigTree theBigTree(bigChain); 

    // create output file 
    TFile * skimFile = new TFile (outputFile, "recreate") ;
    skimFile->cd () ;
    skimTree theSkimTree("HGCALskimmedTree") ;

    Long64_t nEntries = theBigTree.fChain->GetEntries();
    if (nEvents != -1) nEntries = nEvents;
    cout << "** INFO: requested number of events to be skimmed is " << nEntries << endl;

    // ---------------------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------------------
    // LOOP OVER EVENTS

    for (int iEvent = 0; iEvent < nEntries; iEvent++) {
        cout << "- reading entry " << iEvent << endl ;
        
        theSkimTree.clearVars();
        
        int entry_ok = theBigTree.GetEntry(iEvent);
        if(entry_ok<=0) {
            cout << "** WARNING: 0 bits read for event " << theBigTree.event << " (entry " << iEvent << ") - SKIPPING IT" << endl;  
            continue;
        }

        if (DEBUG) cout << "** DEBUG: reading event " << theBigTree.event << endl;

        if (theBigTree.cl3d_n == 0) {
            cout << "** WARNING: no 3D clusters found in the endcaps for event " << theBigTree.event << " (entry " << iEvent << ") - SKIPPING IT" << endl;
            continue;
        }

        theSkimTree.m_run   = theBigTree.run;
        theSkimTree.m_event = theBigTree.event;
        theSkimTree.m_lumi  = theBigTree.lumi;

        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // LOOPS OVER GENERATOR LEVEL INFORMATION

        map<int,int> old2new_tau_idx_map; // map containing < old index of tau , new index of tau after selection >
        map<int,int> TC_iTau_map; // map containing < TC index , tau index >
        map<int,int> cl3d_iTau_map; // map containing < cl3d index , tau index >

        if(isTau){
            int n_gentaus = theBigTree.gentau_pt->size();
            if (DEBUG) cout << "    ** DEBUG: number of the gentaus in the event is " << n_gentaus << endl;

            if (gen3Dmatch) {
                // GENERATOR LEVEL - 3D CLUSTERS MATCHING
                map<int,int> genIdx_iTau_map; // map containing < gen level particle index , gen level tau index from which it came >
                int iTau = -1;
                bool doubleG = false;
                int i_pi_last = n_gentaus;
                int i_k_last = n_gentaus;

                // the entries of gen_pdgid are ordered as follows: [taus, daugthers_of_taus, daughters_of_daughters]
                // the three cases are ordered as follows:
                //              --> tau0, tau1, ... 
                //              --> daughters_of_tau0, daughters_of_tau1, ...
                //              --> the first daughter of each tau is always the tau neutrino, i.e. daughters_of_tau = nu_tau, ...
                //              --> daughters of daughters follow the same scheme
                for (int i_gen=n_gentaus; i_gen < int(theBigTree.gen_pdgid->size()); i_gen++) {
                    // use the nu_tau as separator among daughters
                    if (abs(theBigTree.gen_pdgid->at(i_gen)) == 16) {
                        iTau += 1;
                        genIdx_iTau_map[i_gen] = iTau;
                    }
                    // the gammas are daughters of daughters --> need to retrieve the pi0 where they came from
                    else if (theBigTree.gen_pdgid->at(i_gen) == 22) {
                        if (doubleG) { doubleG = false; continue; }
                        for (int i_pi=i_pi_last; i_pi < i_gen; i_pi++) {
                            if (abs(theBigTree.gen_pdgid->at(i_pi)) == 111) {
                                genIdx_iTau_map[i_gen]   = genIdx_iTau_map[i_pi];
                                genIdx_iTau_map[i_gen+1] = genIdx_iTau_map[i_pi];
                                i_pi_last = i_pi + 1;
                                doubleG = true;
                                break;
                            }
                        }
                    }
                    // the K0L and K0S are daughters of daughters --> need to retrieve the K0 where they came from
                    else if (abs(theBigTree.gen_pdgid->at(i_gen)) == 130 || abs(theBigTree.gen_pdgid->at(i_gen)) == 310) {
                        for (int i_k=i_k_last; i_k < i_gen; i_k++) {
                            if (abs(theBigTree.gen_pdgid->at(i_k)) == 311) {
                                genIdx_iTau_map[i_gen] = genIdx_iTau_map[i_k];
                                i_k_last = i_k + 1;
                                break;
                            }
                        }
                        genIdx_iTau_map[i_gen] = iTau; // if no K0 is found then associate to current iTau
                    }
                    // if the particle is neither a gamma nor a K0L/S we can just save the iTau index
                    else { genIdx_iTau_map[i_gen] = iTau; }
                }

                // now we can match the TCs with the generator level particles
                // map<int,int> TC_pdgID_map; // map containing < TC index , pdgID of the particle that fired the TC >
                for (int i_tc=0; i_tc < theBigTree.tc_n; i_tc++) {
                    if (theBigTree.tc_genparticle_index->at(i_tc) < 0) { continue; }
                    TC_iTau_map[i_tc]  =  genIdx_iTau_map[theBigTree.tc_genparticle_index->at(i_tc)];
                    // TC_pdgID_map[i_tc] =  theBigTree.gen_pdgid->at(theBigTree.tc_genparticle_index->at(i_tc));
                } 

                // now we can match the 3d clusters to the generator level particles
                // map<int,vector<int>> cl3d_pdgID_map; // map containing < cl3d index , pdgID of the particles that fired the cl3d >
                for (int i_cl3d=0; i_cl3d < int(theBigTree.cl3d_id->size()); i_cl3d++) {
                    map<int,int> iTau_occurrence_map; // map containing < iTau , numebr of times it appears in the cluster >
                    // map<int,int> pdgID_occurrence_map; // map containing < generator particle pdgID , numebr of times it appears in the cluster >
                    int iTau_max_occurrence = 0;
                    int iTau_majority_idx = -1;
                    // vector<int> pdgID_occurrence_vec;
                    int total_tcs_in_cl = 0;

                    for (int i_tc=0; i_tc < int(theBigTree.tc_multicluster_id->size()); i_tc++) {
                        if (theBigTree.tc_multicluster_id->at(i_tc) != theBigTree.cl3d_id->at(i_cl3d)) continue; // skip TCs not partaining to the cl3d considered

                        iTau_occurrence_map[TC_iTau_map[i_tc]]   += 1;
                        // pdgID_occurrence_map[TC_pdgID_map[i_tc]] += 1;

                        if (iTau_occurrence_map[TC_iTau_map[i_tc]] > iTau_max_occurrence) {
                            iTau_max_occurrence = iTau_occurrence_map[TC_iTau_map[i_tc]];
                            iTau_majority_idx = TC_iTau_map[i_tc];
                        }

                        total_tcs_in_cl += 1;
                    }

                    // for (map<int,int>::iterator it = pdgID_occurrence_map.begin(); it != pdgID_occurrence_map.end(); it++) {
                    //     // only report a particle to e in a cluster if it accounts for more then 10% of it --> otherwise could just be contamination
                    //     if (float(it->second)/float(total_tcs_in_cl) < 0.05) continue;
                    //     pdgID_occurrence_vec.push_back(it->first);
                    // }

                    cl3d_iTau_map[i_cl3d]  = iTau_majority_idx;
                    // cl3d_pdgID_map[i_cl3d] = pdgID_occurrence_vec;

                    if (float(iTau_max_occurrence)/float(total_tcs_in_cl) < 0.90) {
                        cout << "    ** WARNING: setting 3D cluster - generator tau match with ONLY " << float(iTau_max_occurrence)/float(total_tcs_in_cl)*100 << "% MAJORITY" << endl;
                        if (DEBUG) {
                            for (map<int,int>::iterator it = iTau_occurrence_map.begin(); it != iTau_occurrence_map.end(); it++) { 
                                cout << "           generator tau index=" << it->first << " ; number of occurrencies=" << it->second << endl;
                            }
                            cout << "       total number of TCs in the cluster=" << total_tcs_in_cl << endl;
                        }
                    }
                }
            }

            // BRANCHES FILLING
            int new_tau_idx = 0;
            for (int i_gentau = 0; i_gentau < n_gentaus; i_gentau++){

                if ( abs(theBigTree.gentau_eta->at(i_gentau)) <= 1.5 || abs(theBigTree.gentau_eta->at(i_gentau)) >= 3.0 ) continue;

                bool ishadronic = ( theBigTree.gentau_decayMode->at(i_gentau) == 0 || theBigTree.gentau_decayMode->at(i_gentau) == 1 || theBigTree.gentau_decayMode->at(i_gentau) == 4 || theBigTree.gentau_decayMode->at(i_gentau) == 5 );

                if(!ishadronic) continue;

                theSkimTree.m_gentau_pt.push_back(theBigTree.gentau_pt->at(i_gentau));
                theSkimTree.m_gentau_eta.push_back(theBigTree.gentau_eta->at(i_gentau));
                theSkimTree.m_gentau_phi.push_back(theBigTree.gentau_phi->at(i_gentau));
                theSkimTree.m_gentau_energy.push_back(theBigTree.gentau_energy->at(i_gentau));
                theSkimTree.m_gentau_mass.push_back(theBigTree.gentau_mass->at(i_gentau));
        
                theSkimTree.m_gentau_vis_pt.push_back(theBigTree.gentau_vis_pt->at(i_gentau));
                theSkimTree.m_gentau_vis_eta.push_back(theBigTree.gentau_vis_eta->at(i_gentau));
                theSkimTree.m_gentau_vis_phi.push_back(theBigTree.gentau_vis_phi->at(i_gentau));
                theSkimTree.m_gentau_vis_energy.push_back(theBigTree.gentau_vis_energy->at(i_gentau));
                theSkimTree.m_gentau_vis_mass.push_back(theBigTree.gentau_vis_mass->at(i_gentau));
        
                theSkimTree.m_gentau_products_pt.push_back(theBigTree.gentau_products_pt->at(i_gentau));
                theSkimTree.m_gentau_products_eta.push_back(theBigTree.gentau_products_eta->at(i_gentau));
                theSkimTree.m_gentau_products_phi.push_back(theBigTree.gentau_products_phi->at(i_gentau));
                theSkimTree.m_gentau_products_energy.push_back(theBigTree.gentau_products_energy->at(i_gentau));
                theSkimTree.m_gentau_products_mass.push_back(theBigTree.gentau_products_mass->at(i_gentau));
                theSkimTree.m_gentau_products_id.push_back(theBigTree.gentau_products_id->at(i_gentau));
        
                // here we correct for the idiotic and wrong tau decay mode naming convention used in the bigTrees!!
                int DM = -1;
                if (theBigTree.gentau_decayMode->at(i_gentau) == 0) DM = 0;
                else if (theBigTree.gentau_decayMode->at(i_gentau) == 1) DM = 1;
                else if (theBigTree.gentau_decayMode->at(i_gentau) == 4) DM = 10;
                else if (theBigTree.gentau_decayMode->at(i_gentau) == 5) DM = 11;

                theSkimTree.m_gentau_decayMode.push_back(DM);
                theSkimTree.m_gentau_totNproducts.push_back(theBigTree.gentau_totNproducts->at(i_gentau));
                theSkimTree.m_gentau_totNgamma.push_back(theBigTree.gentau_totNgamma->at(i_gentau));
                theSkimTree.m_gentau_totNpiZero.push_back(theBigTree.gentau_totNpiZero->at(i_gentau));
                theSkimTree.m_gentau_totNcharged.push_back(theBigTree.gentau_totNcharged->at(i_gentau));

                if (gen3Dmatch) {
                    old2new_tau_idx_map[i_gentau] = new_tau_idx;
                    new_tau_idx += 1;
                }
            }

            theSkimTree.m_gentau_n = theSkimTree.m_gentau_pt.size();
            if (DEBUG) cout << "    ** DEBUG: number of selected gentaus in the endcaps is " << theSkimTree.m_gentau_pt.size() << endl;
        }

        else if(isQCD){ 
            int n_genjets = theBigTree.genjet_pt->size();
            if (DEBUG) cout << "    ** DEBUG: number of the genjets in the event is " << n_genjets << endl;

            // GENERATOR LEVEL - 3D CLUSTERS MATCHING


            // BRANCHES FILLING
            for (int i_genjet=0; i_genjet < n_genjets; i_genjet++){

                if ( abs(theBigTree.genjet_eta->at(i_genjet)) <= 1.5 || abs(theBigTree.genjet_eta->at(i_genjet)) >= 3.0 ) continue;
                if ( theBigTree.genjet_pt->at(i_genjet) < 10 || theBigTree.genjet_pt->at(i_genjet) > 500 ) continue;

                TLorentzVector genjet;

                theSkimTree.m_genjet_pt.push_back(theBigTree.genjet_pt->at(i_genjet));
                theSkimTree.m_genjet_eta.push_back(theBigTree.genjet_eta->at(i_genjet));
                theSkimTree.m_genjet_phi.push_back(theBigTree.genjet_phi->at(i_genjet));
                theSkimTree.m_genjet_energy.push_back(theBigTree.genjet_energy->at(i_genjet));
                genjet.SetPtEtaPhiE(theBigTree.genjet_pt->at(i_genjet), theBigTree.genjet_eta->at(i_genjet), theBigTree.genjet_phi->at(i_genjet), theBigTree.genjet_energy->at(i_genjet));
                theSkimTree.m_genjet_mass.push_back(genjet.M());
            }

            theSkimTree.m_genjet_n = theSkimTree.m_genjet_pt.size();
            if (DEBUG) cout << "    ** DEBUG: number of selected genjets in the endcaps is " << theSkimTree.m_genjet_pt.size() << endl;
        }

        if ((isTau && theSkimTree.m_gentau_pt.size() == 0) || (isQCD && theSkimTree.m_genjet_pt.size() == 0)) {
            if (isTau) cout << "** WARNING: no gentaus found in the endcaps for event " << theBigTree.event << " (entry " << iEvent << ") - SKIPPING IT" << endl;
            if (isQCD) cout << "** WARNING: no genjets found in the endcaps for event " << theBigTree.event << " (entry " << iEvent << ") - SKIPPING IT" << endl;            
            continue;
        }

        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // LOOP OVER RECO TRIGGER CELLS

        theSkimTree.m_tc_n = theBigTree.tc_n;
        if (DEBUG) cout << "    ** DEBUG: number of the reco trigger cells in the event is " << theBigTree.tc_n << endl;

        for (int i_tc=0; i_tc < theBigTree.tc_n; i_tc++){
            //if ( abs(theBigTree.tc_eta->at(i_tc)) <= 1.5 || abs(theBigTree.tc_eta->at(i_tc)) >= 3.0 ) continue;

            theSkimTree.m_tc_id.push_back(theBigTree.tc_id->at(i_tc));
            theSkimTree.m_tc_subdet.push_back(theBigTree.tc_subdet->at(i_tc));
            theSkimTree.m_tc_zside.push_back(theBigTree.tc_zside->at(i_tc));
            theSkimTree.m_tc_layer.push_back(theBigTree.tc_layer->at(i_tc));
            theSkimTree.m_tc_waferu.push_back(theBigTree.tc_waferu->at(i_tc));
            theSkimTree.m_tc_waferv.push_back(theBigTree.tc_waferv->at(i_tc));
            theSkimTree.m_tc_wafertype.push_back(theBigTree.tc_wafertype->at(i_tc));
            theSkimTree.m_tc_panel_number.push_back(theBigTree.tc_panel_number->at(i_tc));
            theSkimTree.m_tc_panel_sector.push_back(theBigTree.tc_panel_sector->at(i_tc));
            theSkimTree.m_tc_cellu.push_back(theBigTree.tc_cellu->at(i_tc));
            theSkimTree.m_tc_cellv.push_back(theBigTree.tc_cellv->at(i_tc));
            theSkimTree.m_tc_data.push_back(theBigTree.tc_data->at(i_tc));
            theSkimTree.m_tc_uncompressedCharge.push_back(theBigTree.tc_uncompressedCharge->at(i_tc));
            theSkimTree.m_tc_compressedCharge.push_back(theBigTree.tc_compressedCharge->at(i_tc));
            theSkimTree.m_tc_pt.push_back(theBigTree.tc_pt->at(i_tc));
            theSkimTree.m_tc_mipPt.push_back(theBigTree.tc_mipPt->at(i_tc));
            theSkimTree.m_tc_energy.push_back(theBigTree.tc_energy->at(i_tc));
            theSkimTree.m_tc_eta.push_back(theBigTree.tc_eta->at(i_tc));
            theSkimTree.m_tc_phi.push_back(theBigTree.tc_phi->at(i_tc));
            theSkimTree.m_tc_x.push_back(theBigTree.tc_x->at(i_tc));
            theSkimTree.m_tc_y.push_back(theBigTree.tc_y->at(i_tc));
            theSkimTree.m_tc_z.push_back(theBigTree.tc_z->at(i_tc));
            theSkimTree.m_tc_cluster_id.push_back(theBigTree.tc_cluster_id->at(i_tc));
            theSkimTree.m_tc_multicluster_id.push_back(theBigTree.tc_multicluster_id->at(i_tc));
            theSkimTree.m_tc_multicluster_pt.push_back(theBigTree.tc_multicluster_pt->at(i_tc));

            if (gen3Dmatch) {
                if (TC_iTau_map.find(i_tc) == TC_iTau_map.end()) theSkimTree.m_tc_iTau.push_back(-1);
                else if (old2new_tau_idx_map.find(TC_iTau_map.find(i_tc)->second) == old2new_tau_idx_map.end()) theSkimTree.m_tc_iTau.push_back(-1);
                else theSkimTree.m_tc_iTau.push_back(old2new_tau_idx_map.find(TC_iTau_map.find(i_tc)->second)->second);
            }

            //theSkimTree.m_tc_pdgid.push_back(TC_pdgID_map[i_tc]);
        }

        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // LOOP OVER RECO 3D CLUSTERS

        theSkimTree.m_cl3d_n = theBigTree.cl3d_n;
        if (DEBUG) cout << "    ** DEBUG: number of the reco 3d clusters in the event is " << theBigTree.cl3d_n << endl;

        for (int i_cl3d=0; i_cl3d < theBigTree.cl3d_n; i_cl3d++){
            //if ( abs(theBigTree.cl3d_eta->at(i_cl3d)) <= 1.5 || abs(theBigTree.cl3d_eta->at(i_cl3d)) >= 3.0 ) continue;

            theSkimTree.m_cl3d_id.push_back(theBigTree.cl3d_id->at(i_cl3d));
            theSkimTree.m_cl3d_pt.push_back(theBigTree.cl3d_pt->at(i_cl3d));
            theSkimTree.m_cl3d_energy.push_back(theBigTree.cl3d_energy->at(i_cl3d));
            theSkimTree.m_cl3d_eta.push_back(theBigTree.cl3d_eta->at(i_cl3d));
            theSkimTree.m_cl3d_phi.push_back(theBigTree.cl3d_phi->at(i_cl3d));
            theSkimTree.m_cl3d_clusters_n.push_back(theBigTree.cl3d_clusters_n->at(i_cl3d));
            theSkimTree.m_cl3d_clusters_id.push_back(theBigTree.cl3d_clusters_id->at(i_cl3d));
            theSkimTree.m_cl3d_layer_pt.push_back(theBigTree.cl3d_layer_pt->at(i_cl3d));
            theSkimTree.m_cl3d_showerlength.push_back(theBigTree.cl3d_showerlength->at(i_cl3d));
            theSkimTree.m_cl3d_coreshowerlength.push_back(theBigTree.cl3d_coreshowerlength->at(i_cl3d));
            theSkimTree.m_cl3d_firstlayer.push_back(theBigTree.cl3d_firstlayer->at(i_cl3d));
            theSkimTree.m_cl3d_maxlayer.push_back(theBigTree.cl3d_maxlayer->at(i_cl3d));     
            theSkimTree.m_cl3d_seetot.push_back(theBigTree.cl3d_seetot->at(i_cl3d));
            theSkimTree.m_cl3d_seemax.push_back(theBigTree.cl3d_seemax->at(i_cl3d));
            theSkimTree.m_cl3d_spptot.push_back(theBigTree.cl3d_spptot->at(i_cl3d));
            theSkimTree.m_cl3d_sppmax.push_back(theBigTree.cl3d_sppmax->at(i_cl3d));
            theSkimTree.m_cl3d_szz.push_back(theBigTree.cl3d_szz->at(i_cl3d));
            theSkimTree.m_cl3d_srrtot.push_back(theBigTree.cl3d_srrtot->at(i_cl3d));
            theSkimTree.m_cl3d_srrmax.push_back(theBigTree.cl3d_srrmax->at(i_cl3d));
            theSkimTree.m_cl3d_srrmean.push_back(theBigTree.cl3d_srrmean->at(i_cl3d));
            theSkimTree.m_cl3d_emaxe.push_back(theBigTree.cl3d_emaxe->at(i_cl3d));
            theSkimTree.m_cl3d_hoe.push_back(theBigTree.cl3d_hoe->at(i_cl3d));
            theSkimTree.m_cl3d_meanz.push_back(theBigTree.cl3d_meanz->at(i_cl3d));
            theSkimTree.m_cl3d_layer10.push_back(theBigTree.cl3d_layer10->at(i_cl3d));
            theSkimTree.m_cl3d_layer50.push_back(theBigTree.cl3d_layer50->at(i_cl3d));
            theSkimTree.m_cl3d_layer90.push_back(theBigTree.cl3d_layer90->at(i_cl3d));
            theSkimTree.m_cl3d_ntc67.push_back(theBigTree.cl3d_ntc67->at(i_cl3d));
            theSkimTree.m_cl3d_ntc90.push_back(theBigTree.cl3d_ntc90->at(i_cl3d));
            theSkimTree.m_cl3d_bdteg.push_back(theBigTree.cl3d_bdteg->at(i_cl3d));
            theSkimTree.m_cl3d_quality.push_back(theBigTree.cl3d_quality->at(i_cl3d));

            if (gen3Dmatch) {
                if (cl3d_iTau_map.find(i_cl3d) == cl3d_iTau_map.end()) theSkimTree.m_cl3d_iTau.push_back(-1);
                else if (old2new_tau_idx_map.find(cl3d_iTau_map.find(i_cl3d)->second) == old2new_tau_idx_map.end()) theSkimTree.m_cl3d_iTau.push_back(-1);
                else theSkimTree.m_cl3d_iTau.push_back(old2new_tau_idx_map.find(cl3d_iTau_map.find(i_cl3d)->second)->second);
            }

            //theSkimTree.m_cl3d_pdgid.push_back(cl3d_pdgID_map[i_cl3d]);
        }

        // ---------------------------------------------------------------------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        // LOOP OVER RECO TOWERS

        theSkimTree.m_tower_n = theBigTree.tower_n;
        if (DEBUG) cout << "    ** DEBUG: number of the reco towers in the event is " << theBigTree.cl3d_n << endl;

        for (int i_tower=0; i_tower < theBigTree.tower_n; i_tower++){
            theSkimTree.m_tower_pt.push_back(theBigTree.tower_pt->at(i_tower));
            theSkimTree.m_tower_energy.push_back(theBigTree.tower_energy->at(i_tower));
            theSkimTree.m_tower_eta.push_back(theBigTree.tower_eta->at(i_tower));
            theSkimTree.m_tower_phi.push_back(theBigTree.tower_phi->at(i_tower));
            theSkimTree.m_tower_etEm.push_back(theBigTree.tower_etEm->at(i_tower));
            theSkimTree.m_tower_etHad.push_back(theBigTree.tower_etHad->at(i_tower));
            theSkimTree.m_tower_iEta.push_back(theBigTree.tower_iEta->at(i_tower));
            theSkimTree.m_tower_iPhi.push_back(theBigTree.tower_iPhi->at(i_tower));
        }

        theSkimTree.Fill();
    }

    skimFile->cd();
    theSkimTree.Write();
    skimFile->Write();
    skimFile->Close();

    cout << "... SKIM finished, exiting." << endl;

    return 0;
}