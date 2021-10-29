#ifndef skimTree_h
#define skimTree_h

#include "TTree.h"
#include "TString.h"

#include <vector>

using namespace std;

// create the small tree with the branches I am interested in
struct skimTree
{
    skimTree(TString treeName) : m_skimT (new TTree (treeName, "skimmed tree for HGCAL studies")) {
        init () ;
    }

    int Fill() {
        return m_skimT->Fill ();
    }

    int Write() {
        return m_skimT->Write();
    }

    int clearVars() {
        // GENERAL INFORMATION
        m_run = -1. ;
        m_event = -1. ;
        m_lumi = -1. ;

        // GENERATOR LEVEL INFORMATION
        m_genjet_n = -1.;
        m_genjet_pt.clear();
        m_genjet_eta.clear();
        m_genjet_phi.clear();
        m_genjet_energy.clear();
        m_genjet_mass.clear();

        m_gentau_n = -1.;
        m_gentau_pt.clear();
        m_gentau_eta.clear();
        m_gentau_phi.clear();
        m_gentau_energy.clear();
        m_gentau_mass.clear();

        m_gentau_vis_pt.clear();
        m_gentau_vis_eta.clear();
        m_gentau_vis_phi.clear();
        m_gentau_vis_energy.clear();
        m_gentau_vis_mass.clear();
        m_gentau_decayMode.clear();

        m_gentau_products_pt.clear();
        m_gentau_products_eta.clear();
        m_gentau_products_phi.clear();
        m_gentau_products_energy.clear();
        m_gentau_products_mass.clear();
        m_gentau_products_id.clear();
        m_gentau_totNproducts.clear();
        m_gentau_totNgamma.clear();
        m_gentau_totNpiZero.clear();
        m_gentau_totNcharged.clear();

        // RECO TRIGGER CELLS INFORMATION
/*        m_tc_n = -1.;
        m_tc_id.clear();
        m_tc_subdet.clear();
        m_tc_zside.clear();
        m_tc_layer.clear();
        m_tc_waferu.clear();
        m_tc_waferv.clear();
        m_tc_wafertype.clear();
        m_tc_panel_number.clear();
        m_tc_panel_sector.clear();
        m_tc_cellu.clear();
        m_tc_cellv.clear();
        m_tc_data.clear();
        m_tc_uncompressedCharge.clear();
        m_tc_compressedCharge.clear();
        m_tc_pt.clear();
        m_tc_mipPt.clear();
        m_tc_energy.clear();
        m_tc_eta.clear();
        m_tc_phi.clear();
        m_tc_x.clear();
        m_tc_y.clear();
        m_tc_z.clear();
        m_tc_cluster_id.clear();
        m_tc_multicluster_id.clear();
        m_tc_multicluster_pt.clear();
        m_tc_iTau.clear();
        m_tc_pdgid.clear();
*/
        // RECO 3D CLUSTERS INFORMATION
        m_cl3d_n = -1.;
        m_cl3d_id.clear();
        m_cl3d_pt.clear();
        m_cl3d_energy.clear();
        m_cl3d_eta.clear();
        m_cl3d_phi.clear();
        m_cl3d_clusters_n.clear();
        m_cl3d_clusters_id.clear();
        m_cl3d_layer_pt.clear();
        m_cl3d_showerlength.clear();
        m_cl3d_coreshowerlength.clear();
        m_cl3d_firstlayer.clear();
        m_cl3d_maxlayer.clear();
        m_cl3d_seetot.clear();
        m_cl3d_seemax.clear();
        m_cl3d_spptot.clear();
        m_cl3d_sppmax.clear();
        m_cl3d_szz.clear();
        m_cl3d_srrtot.clear();
        m_cl3d_srrmax.clear();
        m_cl3d_srrmean.clear();
        m_cl3d_emaxe.clear();
        m_cl3d_hoe.clear();
        m_cl3d_meanz.clear();
        m_cl3d_layer10.clear();
        m_cl3d_layer50.clear();
        m_cl3d_layer90.clear();
        m_cl3d_ntc67.clear();
        m_cl3d_ntc90.clear();
        m_cl3d_bdteg.clear();
        m_cl3d_quality.clear();
        m_cl3d_iTau.clear();
        m_cl3d_pdgid.clear();

        // RECO TOWER INFORMATION
        m_tower_n = -1.;
        m_tower_pt.clear();
        m_tower_energy.clear();
        m_tower_eta.clear();
        m_tower_phi.clear();
        m_tower_etEm.clear();
        m_tower_etHad.clear();
        m_tower_iEta.clear();
        m_tower_iPhi.clear();


        return 0;
    }

    int init() {
        // GENERAL INFORMATION
        m_skimT->Branch("run", &m_run);
        m_skimT->Branch("event", &m_event);
        m_skimT->Branch("lumi", &m_lumi);

        // GENERATOR LEVEL INFORMATION
        m_skimT->Branch("genjet_n", &m_genjet_n);
        m_skimT->Branch("genjet_pt", &m_genjet_pt);
        m_skimT->Branch("genjet_eta", &m_genjet_eta);
        m_skimT->Branch("genjet_phi", &m_genjet_phi);
        m_skimT->Branch("genjet_energy", &m_genjet_energy);
        m_skimT->Branch("genjet_mass", &m_genjet_mass);

        m_skimT->Branch("gentau_n", &m_gentau_n);
        m_skimT->Branch("gentau_pt", &m_gentau_pt);
        m_skimT->Branch("gentau_eta", &m_gentau_eta);
        m_skimT->Branch("gentau_phi", &m_gentau_phi);
        m_skimT->Branch("gentau_energy", &m_gentau_energy);
        m_skimT->Branch("gentau_mass", &m_gentau_mass);

        m_skimT->Branch("gentau_vis_pt", &m_gentau_vis_pt);
        m_skimT->Branch("gentau_vis_eta", &m_gentau_vis_eta);
        m_skimT->Branch("gentau_vis_phi", &m_gentau_vis_phi);
        m_skimT->Branch("gentau_vis_energy", &m_gentau_vis_energy);
        m_skimT->Branch("gentau_vis_mass", &m_gentau_vis_mass);
        m_skimT->Branch("gentau_decayMode", &m_gentau_decayMode);

        m_skimT->Branch("gentau_products_pt", &m_gentau_products_pt);
        m_skimT->Branch("gentau_products_eta", &m_gentau_products_eta);
        m_skimT->Branch("gentau_products_phi", &m_gentau_products_phi);
        m_skimT->Branch("gentau_products_energy", &m_gentau_products_energy);
        m_skimT->Branch("gentau_products_mass", &m_gentau_products_mass);
        m_skimT->Branch("gentau_products_id", &m_gentau_products_id);
        m_skimT->Branch("gentau_totNproducts", &m_gentau_totNproducts);
        m_skimT->Branch("gentau_totNgamma", &m_gentau_totNgamma);
        m_skimT->Branch("gentau_totNpiZero", &m_gentau_totNpiZero);
        m_skimT->Branch("gentau_totNcharged", &m_gentau_totNcharged);

        // RECO TRIGGER CELLS INFORMATION
/*        m_skimT->Branch("tc_n", &m_tc_n);
        m_skimT->Branch("tc_id", &m_tc_id);
        m_skimT->Branch("tc_subdet", &m_tc_subdet);
        m_skimT->Branch("tc_zside", &m_tc_zside);
        m_skimT->Branch("tc_layer", &m_tc_layer);
        m_skimT->Branch("tc_waferu", &m_tc_waferu);
        m_skimT->Branch("tc_waferv", &m_tc_waferv);
        m_skimT->Branch("tc_wafertype", &m_tc_wafertype);
        m_skimT->Branch("tc_panel_number", &m_tc_panel_number);
        m_skimT->Branch("tc_panel_sector", &m_tc_panel_sector);
        m_skimT->Branch("tc_cellu", &m_tc_cellu);
        m_skimT->Branch("tc_cellv", &m_tc_cellv);
        m_skimT->Branch("tc_data", &m_tc_data);
        m_skimT->Branch("tc_uncompressedCharge", &m_tc_uncompressedCharge);
        m_skimT->Branch("tc_compressedCharge", &m_tc_compressedCharge);
        m_skimT->Branch("tc_pt", &m_tc_pt);
        m_skimT->Branch("tc_mipPt", &m_tc_mipPt);
        m_skimT->Branch("tc_energy", &m_tc_energy);
        m_skimT->Branch("tc_eta", &m_tc_eta);
        m_skimT->Branch("tc_phi", &m_tc_phi);
        m_skimT->Branch("tc_x", &m_tc_x);
        m_skimT->Branch("tc_y", &m_tc_y);
        m_skimT->Branch("tc_z", &m_tc_z);
        m_skimT->Branch("tc_cluster_id", &m_tc_cluster_id);
        m_skimT->Branch("tc_multicluster_id", &m_tc_multicluster_id);
        m_skimT->Branch("tc_multicluster_pt", &m_tc_multicluster_pt);
        m_skimT->Branch("tc_iTau", &m_tc_iTau);
        m_skimT->Branch("tc_pdgid", &m_tc_pdgid);
*/
        // RECO 3D CLUSTERS INFORMATION
        m_skimT->Branch("cl3d_n", &m_cl3d_n);
        m_skimT->Branch("cl3d_id", &m_cl3d_id);
        m_skimT->Branch("cl3d_pt", &m_cl3d_pt);
        m_skimT->Branch("cl3d_energy", &m_cl3d_energy);
        m_skimT->Branch("cl3d_eta", &m_cl3d_eta);
        m_skimT->Branch("cl3d_phi", &m_cl3d_phi);
        m_skimT->Branch("cl3d_clusters_n", &m_cl3d_clusters_n);
        m_skimT->Branch("cl3d_clusters_id", &m_cl3d_clusters_id);
        m_skimT->Branch("cl3d_layer_pt", &m_cl3d_layer_pt);
        m_skimT->Branch("cl3d_showerlength", &m_cl3d_showerlength);
        m_skimT->Branch("cl3d_coreshowerlength", &m_cl3d_coreshowerlength);
        m_skimT->Branch("cl3d_firstlayer", &m_cl3d_firstlayer);
        m_skimT->Branch("cl3d_maxlayer", &m_cl3d_maxlayer);
        m_skimT->Branch("cl3d_seetot", &m_cl3d_seetot);
        m_skimT->Branch("cl3d_seemax", &m_cl3d_seemax);
        m_skimT->Branch("cl3d_spptot", &m_cl3d_spptot);
        m_skimT->Branch("cl3d_sppmax", &m_cl3d_sppmax);
        m_skimT->Branch("cl3d_szz", &m_cl3d_szz);
        m_skimT->Branch("cl3d_srrtot", &m_cl3d_srrtot);
        m_skimT->Branch("cl3d_srrmax", &m_cl3d_srrmax);
        m_skimT->Branch("cl3d_srrmean", &m_cl3d_srrmean);
        m_skimT->Branch("cl3d_emaxe", &m_cl3d_emaxe);
        m_skimT->Branch("cl3d_hoe", &m_cl3d_hoe);
        m_skimT->Branch("cl3d_meanz", &m_cl3d_meanz);
        m_skimT->Branch("cl3d_layer10", &m_cl3d_layer10);
        m_skimT->Branch("cl3d_layer50", &m_cl3d_layer50);
        m_skimT->Branch("cl3d_layer90", &m_cl3d_layer90);
        m_skimT->Branch("cl3d_ntc67", &m_cl3d_ntc67);
        m_skimT->Branch("cl3d_ntc90", &m_cl3d_ntc90);
        m_skimT->Branch("cl3d_bdteg", &m_cl3d_bdteg);
        m_skimT->Branch("cl3d_quality", &m_cl3d_quality);
        m_skimT->Branch("cl3d_iTau", &m_cl3d_iTau);
        m_skimT->Branch("cl3d_pdgid", &m_cl3d_pdgid);

        // RECO TOWER INFORMATION
        m_skimT->Branch("tower_n", &m_tower_n);
        m_skimT->Branch("tower_pt", &m_tower_pt);
        m_skimT->Branch("tower_energy", &m_tower_energy);
        m_skimT->Branch("tower_eta", &m_tower_eta);
        m_skimT->Branch("tower_phi", &m_tower_phi);
        m_skimT->Branch("tower_etEm", &m_tower_etEm);
        m_skimT->Branch("tower_etHad", &m_tower_etHad);
        m_skimT->Branch("tower_iEta", &m_tower_iEta);
        m_skimT->Branch("tower_iPhi", &m_tower_iPhi);

        return 0;
    }

    // THE TTREE ITSELF
    TTree * m_skimT ;

    //---------------------------------------
    // VARIABLES THAT GO INSIDE THE TTREE

    // GENERAL INFORMATION
    int m_run;
    int m_event;
    int m_lumi;

    // GENERATOR LEVEL INFORMATION
    int              m_genjet_n;
    vector<float>    m_genjet_pt; 
    vector<float>    m_genjet_eta;
    vector<float>    m_genjet_phi;
    vector<float>    m_genjet_energy;
    vector<float>    m_genjet_mass;

    int              m_gentau_n;
    vector<float>    m_gentau_pt; 
    vector<float>    m_gentau_eta;
    vector<float>    m_gentau_phi;
    vector<float>    m_gentau_energy;
    vector<float>    m_gentau_mass;

    vector<float>    m_gentau_vis_pt; 
    vector<float>    m_gentau_vis_eta;
    vector<float>    m_gentau_vis_phi;
    vector<float>    m_gentau_vis_energy;
    vector<float>    m_gentau_vis_mass;
    vector<int>      m_gentau_decayMode;
    vector<vector<float>>    m_gentau_products_pt;
    vector<vector<float>>    m_gentau_products_eta;
    vector<vector<float>>    m_gentau_products_phi;
    vector<vector<float>>    m_gentau_products_energy;
    vector<vector<float>>    m_gentau_products_mass;
    vector<vector<int>>      m_gentau_products_id;

    vector<int>    m_gentau_totNproducts;
    vector<int>    m_gentau_totNgamma;
    vector<int>    m_gentau_totNpiZero;
    vector<int>    m_gentau_totNcharged;

    // RECO TRIGGER CELLS INFORMATION
/*    int                     m_tc_n;
    vector<unsigned int>    m_tc_id;
    vector<int>             m_tc_subdet;
    vector<int>             m_tc_zside;
    vector<int>             m_tc_layer;
    vector<int>             m_tc_waferu;
    vector<int>             m_tc_waferv;
    vector<int>             m_tc_wafertype;
    vector<int>             m_tc_panel_number;
    vector<int>             m_tc_panel_sector;
    vector<int>             m_tc_cellu;
    vector<int>             m_tc_cellv;  
    vector<unsigned int>    m_tc_data;
    vector<unsigned int>    m_tc_uncompressedCharge;
    vector<unsigned int>    m_tc_compressedCharge;
    vector<float>           m_tc_pt;
    vector<float>           m_tc_mipPt;
    vector<float>           m_tc_energy;
    vector<float>           m_tc_eta;
    vector<float>           m_tc_phi;
    vector<float>           m_tc_x;
    vector<float>           m_tc_y;
    vector<float>           m_tc_z;
    vector<unsigned int>    m_tc_cluster_id;
    vector<unsigned int>    m_tc_multicluster_id;
    vector<float>           m_tc_multicluster_pt;
    vector<int>             m_tc_iTau;
    vector<int>             m_tc_pdgid;
*/
    // RECO 3D CLUSTERS INFORMATION
    int                             m_cl3d_n;
    vector<unsigned int>            m_cl3d_id;
    vector<float>                   m_cl3d_pt;
    vector<float>                   m_cl3d_energy;
    vector<float>                   m_cl3d_eta;
    vector<float>                   m_cl3d_phi;
    vector<int>                     m_cl3d_clusters_n;
    vector<vector<unsigned int>>    m_cl3d_clusters_id;
    vector<vector<float>>           m_cl3d_layer_pt;
    vector<int>                     m_cl3d_showerlength;
    vector<int>                     m_cl3d_coreshowerlength;
    vector<int>                     m_cl3d_firstlayer;
    vector<int>                     m_cl3d_maxlayer;
    vector<float>                   m_cl3d_seetot;
    vector<float>                   m_cl3d_seemax;
    vector<float>                   m_cl3d_spptot;
    vector<float>                   m_cl3d_sppmax;
    vector<float>                   m_cl3d_szz;
    vector<float>                   m_cl3d_srrtot;
    vector<float>                   m_cl3d_srrmax;
    vector<float>                   m_cl3d_srrmean;
    vector<float>                   m_cl3d_emaxe;
    vector<float>                   m_cl3d_hoe;
    vector<float>                   m_cl3d_meanz;
    vector<float>                   m_cl3d_layer10;
    vector<float>                   m_cl3d_layer50;
    vector<float>                   m_cl3d_layer90;
    vector<float>                   m_cl3d_ntc67;
    vector<float>                   m_cl3d_ntc90;
    vector<float>                   m_cl3d_bdteg;
    vector<int>                     m_cl3d_quality;
    vector<int>                     m_cl3d_iTau;
    vector<vector<int>>             m_cl3d_pdgid;

    // RECO TOWER INFORMATION
    int             m_tower_n;
    vector<float>   m_tower_pt;
    vector<float>   m_tower_energy;
    vector<float>   m_tower_eta;
    vector<float>   m_tower_phi;
    vector<float>   m_tower_etEm;
    vector<float>   m_tower_etHad;
    vector<int>     m_tower_iEta;
    vector<int>     m_tower_iPhi;
};

#endif