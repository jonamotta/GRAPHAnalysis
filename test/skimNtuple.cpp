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
           << " inputFileNameList outputFileName nEvents isTau isQCD DEBUG" << endl ;
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

    bool DEBUG = false;
    string opt20 (argv[6]);
    if (opt20 == "1") DEBUG = true;

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

        theSkimTree.m_run   = theBigTree.run;
        theSkimTree.m_event = theBigTree.event;
        theSkimTree.m_lumi  = theBigTree.lumi;

        // LOOP OVER GENERATOR LEVEL INFORMATION
        if(isTau){
            int n_gentaus = theBigTree.gentau_pt->size();
            if (DEBUG) cout << "    ** DEBUG: number of the gentaus in the event is " << n_gentaus << endl;

            for (int i_gentau = 0; i_gentau < n_gentaus; i_gentau++){

                if ( abs(theBigTree.gentau_eta->at(i_gentau)) <= 1.5 || abs(theBigTree.gentau_eta->at(i_gentau)) >= 3.0 ) continue;

                bool ishadronic = ( theBigTree.gentau_decayMode->at(i_gentau) == 0 || theBigTree.gentau_decayMode->at(i_gentau) == 1 || theBigTree.gentau_decayMode->at(i_gentau) == 10 || theBigTree.gentau_decayMode->at(i_gentau) == 11 );

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
        
                theSkimTree.m_gentau_decayMode.push_back(theBigTree.gentau_decayMode->at(i_gentau));
                theSkimTree.m_gentau_totNproducts.push_back(theBigTree.gentau_totNproducts->at(i_gentau));
                theSkimTree.m_gentau_totNgamma.push_back(theBigTree.gentau_totNgamma->at(i_gentau));
                theSkimTree.m_gentau_totNpiZero.push_back(theBigTree.gentau_totNpiZero->at(i_gentau));
                theSkimTree.m_gentau_totNcharged.push_back(theBigTree.gentau_totNcharged->at(i_gentau));

                theSkimTree.m_genpart_gen.push_back(theBigTree.genpart_gen->at(i_gentau));
            }

            theSkimTree.m_gentau_n = theSkimTree.m_gentau_pt.size();
            if (DEBUG) cout << "    ** DEBUG: number of selected gentaus in the endcaps is " << theSkimTree.m_gentau_pt.size() << endl;
        }

        else if(isQCD){ 
            int n_genjets = theBigTree.genjet_pt->size();
            if (DEBUG) cout << "    ** DEBUG: number of the genjets in the event is " << n_genjets << endl;

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

                theSkimTree.m_genpart_gen.push_back(theBigTree.genpart_gen->at(i_genjet)); 
            }

            theSkimTree.m_genjet_n = theSkimTree.m_genjet_pt.size();
            if (DEBUG) cout << "    ** DEBUG: number of selected genjets in the endcaps is " << theSkimTree.m_genjet_pt.size() << endl;
        }

        if ((isTau && theSkimTree.m_gentau_pt.size() == 0) || (isQCD && theSkimTree.m_genjet_pt.size() == 0)) {
            if (isTau) cout << "** WARNING: no gentaus found in the endcaps for event " << theBigTree.event << " (entry " << iEvent << ") - SKIPPING IT" << endl;
            if (isQCD) cout << "** WARNING: no genjets found in the endcaps for event " << theBigTree.event << " (entry " << iEvent << ") - SKIPPING IT" << endl;            
            continue;
        }


        // SET GENPARTICLE FLAGS FOR THE 3D CLUSTERS
        for (unsigned int i_cl3d=0; i_cl3d < theBigTree.cl3d_id->size(); i_cl3d++) {
            map<int,int> idx_occurrence_map;
            float max_occurrence = 0;
            int majority_idx = -1;
            float total_tcs_in_cl = 0;

            for (unsigned int j=0; j < theBigTree.tc_multicluster_id->size(); j++) {
                int gen_idx = theBigTree.tc_genparticle_index->at(j);

                if (theBigTree.tc_multicluster_id->at(j) == theBigTree.cl3d_id->at(i_cl3d)) {
                    idx_occurrence_map[gen_idx] += 1;
                    total_tcs_in_cl += 1;
                    
                    if (idx_occurrence_map[gen_idx] > max_occurrence) {
                        max_occurrence = idx_occurrence_map[gen_idx]; 
                        majority_idx = gen_idx;
                    }
                }
            }

            theSkimTree.m_cl3d_genparticle_index.push_back(majority_idx);

            if (max_occurrence/total_tcs_in_cl < 0.50) {
                cout << "    ** WARNING: set genparticle index for 3D cluster with ONLY " << max_occurrence/total_tcs_in_cl*100 << "% MAJORITY" << endl;
                for (map<int,int>::iterator it = idx_occurrence_map.begin(); it != idx_occurrence_map.end(); it++) { 
                    cout << "           genparticle_index=" << it->first << " ; number of occurrencies=" << it->second << endl;
                }
            }
        }

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
            theSkimTree.m_tc_genparticle_index.push_back(theBigTree.tc_genparticle_index->at(i_tc));
        }

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
        }

        // LOOP OVER CALO-TRUTH TRIGGER CELLS
        theSkimTree.m_tctruth_n = theBigTree.tctruth_n;
        if (DEBUG) cout << "    ** DEBUG: number of the calo-truth trigger cells in the event is " << theBigTree.tctruth_n << endl;

        for (int i_tc=0; i_tc < theBigTree.tctruth_n; i_tc++){
            //if ( abs(theBigTree.tctruth_eta->at(i_tc)) <= 1.5 || abs(theBigTree.tctruth_eta->at(i_tc)) >= 3.0 ) continue;

            theSkimTree.m_tctruth_id.push_back(theBigTree.tctruth_id->at(i_tc));
            theSkimTree.m_tctruth_subdet.push_back(theBigTree.tctruth_subdet->at(i_tc));
            theSkimTree.m_tctruth_zside.push_back(theBigTree.tctruth_zside->at(i_tc));
            theSkimTree.m_tctruth_layer.push_back(theBigTree.tctruth_layer->at(i_tc));
            theSkimTree.m_tctruth_waferu.push_back(theBigTree.tctruth_waferu->at(i_tc));
            theSkimTree.m_tctruth_waferv.push_back(theBigTree.tctruth_waferv->at(i_tc));
            theSkimTree.m_tctruth_wafertype.push_back(theBigTree.tctruth_wafertype->at(i_tc));
            theSkimTree.m_tctruth_panel_number.push_back(theBigTree.tctruth_panel_number->at(i_tc));
            theSkimTree.m_tctruth_panel_sector.push_back(theBigTree.tctruth_panel_sector->at(i_tc));
            theSkimTree.m_tctruth_cellu.push_back(theBigTree.tctruth_cellu->at(i_tc));
            theSkimTree.m_tctruth_cellv.push_back(theBigTree.tctruth_cellv->at(i_tc));
            theSkimTree.m_tctruth_data.push_back(theBigTree.tctruth_data->at(i_tc));
            theSkimTree.m_tctruth_uncompressedCharge.push_back(theBigTree.tctruth_uncompressedCharge->at(i_tc));
            theSkimTree.m_tctruth_compressedCharge.push_back(theBigTree.tctruth_compressedCharge->at(i_tc));
            theSkimTree.m_tctruth_pt.push_back(theBigTree.tctruth_pt->at(i_tc));
            theSkimTree.m_tctruth_mipPt.push_back(theBigTree.tctruth_mipPt->at(i_tc));
            theSkimTree.m_tctruth_energy.push_back(theBigTree.tctruth_energy->at(i_tc));
            theSkimTree.m_tctruth_eta.push_back(theBigTree.tctruth_eta->at(i_tc));
            theSkimTree.m_tctruth_phi.push_back(theBigTree.tctruth_phi->at(i_tc));
            theSkimTree.m_tctruth_x.push_back(theBigTree.tctruth_x->at(i_tc));
            theSkimTree.m_tctruth_y.push_back(theBigTree.tctruth_y->at(i_tc));
            theSkimTree.m_tctruth_z.push_back(theBigTree.tctruth_z->at(i_tc));
            theSkimTree.m_tctruth_cluster_id.push_back(theBigTree.tctruth_cluster_id->at(i_tc));
            theSkimTree.m_tctruth_multicluster_id.push_back(theBigTree.tctruth_multicluster_id->at(i_tc));
            theSkimTree.m_tctruth_multicluster_pt.push_back(theBigTree.tctruth_multicluster_pt->at(i_tc));
        }

        // LOOP OVER RECO 3D CLUSTERS
        theSkimTree.m_cl3dtruth_n = theBigTree.cl3dtruth_n;
        if (DEBUG) cout << "    ** DEBUG: number of the calo-truth clusters in the event is " << theBigTree.cl3dtruth_n << endl;

        for (int i_cl3d = 0; i_cl3d < theBigTree.cl3dtruth_n; i_cl3d++){    
            //if ( abs(theBigTree.cl3dtruth_eta->at(i_cl3d)) <= 1.5 || abs(theBigTree.cl3dtruth_eta->at(i_cl3d)) >= 3.0 ) continue;

            theSkimTree.m_cl3dtruth_id.push_back(theBigTree.cl3dtruth_id->at(i_cl3d));
            theSkimTree.m_cl3dtruth_pt.push_back(theBigTree.cl3dtruth_pt->at(i_cl3d));
            theSkimTree.m_cl3dtruth_energy.push_back(theBigTree.cl3dtruth_energy->at(i_cl3d));
            theSkimTree.m_cl3dtruth_eta.push_back(theBigTree.cl3dtruth_eta->at(i_cl3d));
            theSkimTree.m_cl3dtruth_phi.push_back(theBigTree.cl3dtruth_phi->at(i_cl3d));
            theSkimTree.m_cl3dtruth_clusters_n.push_back(theBigTree.cl3dtruth_clusters_n->at(i_cl3d));
            theSkimTree.m_cl3dtruth_clusters_id.push_back(theBigTree.cl3dtruth_clusters_id->at(i_cl3d));
            theSkimTree.m_cl3dtruth_showerlength.push_back(theBigTree.cl3dtruth_showerlength->at(i_cl3d));
            theSkimTree.m_cl3dtruth_coreshowerlength.push_back(theBigTree.cl3dtruth_coreshowerlength->at(i_cl3d));
            theSkimTree.m_cl3dtruth_firstlayer.push_back(theBigTree.cl3dtruth_firstlayer->at(i_cl3d));
            theSkimTree.m_cl3dtruth_maxlayer.push_back(theBigTree.cl3dtruth_maxlayer->at(i_cl3d));     
            theSkimTree.m_cl3dtruth_seetot.push_back(theBigTree.cl3dtruth_seetot->at(i_cl3d));
            theSkimTree.m_cl3dtruth_seemax.push_back(theBigTree.cl3dtruth_seemax->at(i_cl3d));
            theSkimTree.m_cl3dtruth_spptot.push_back(theBigTree.cl3dtruth_spptot->at(i_cl3d));
            theSkimTree.m_cl3dtruth_sppmax.push_back(theBigTree.cl3dtruth_sppmax->at(i_cl3d));
            theSkimTree.m_cl3dtruth_szz.push_back(theBigTree.cl3dtruth_szz->at(i_cl3d));
            theSkimTree.m_cl3dtruth_srrtot.push_back(theBigTree.cl3dtruth_srrtot->at(i_cl3d));
            theSkimTree.m_cl3dtruth_srrmax.push_back(theBigTree.cl3dtruth_srrmax->at(i_cl3d));
            theSkimTree.m_cl3dtruth_srrmean.push_back(theBigTree.cl3dtruth_srrmean->at(i_cl3d));
            theSkimTree.m_cl3dtruth_emaxe.push_back(theBigTree.cl3dtruth_emaxe->at(i_cl3d));
            theSkimTree.m_cl3dtruth_hoe.push_back(theBigTree.cl3dtruth_hoe->at(i_cl3d));
            theSkimTree.m_cl3dtruth_meanz.push_back(theBigTree.cl3dtruth_meanz->at(i_cl3d));
            theSkimTree.m_cl3dtruth_layer10.push_back(theBigTree.cl3dtruth_layer10->at(i_cl3d));
            theSkimTree.m_cl3dtruth_layer50.push_back(theBigTree.cl3dtruth_layer50->at(i_cl3d));
            theSkimTree.m_cl3dtruth_layer90.push_back(theBigTree.cl3dtruth_layer90->at(i_cl3d));
            theSkimTree.m_cl3dtruth_ntc67.push_back(theBigTree.cl3dtruth_ntc67->at(i_cl3d));
            theSkimTree.m_cl3dtruth_ntc90.push_back(theBigTree.cl3dtruth_ntc90->at(i_cl3d));
            theSkimTree.m_cl3dtruth_bdteg.push_back(theBigTree.cl3dtruth_bdteg->at(i_cl3d));
            theSkimTree.m_cl3dtruth_quality.push_back(theBigTree.cl3dtruth_quality->at(i_cl3d));
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