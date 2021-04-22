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

        m_genpart_gen.clear();

        // RECO TRIGGER CELLS INFORMATION
        m_tc_n = -1.;
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
        m_tc_genparticle_index.clear();

        // RECO 3D CLUSTERS INFORMATION
        m_cl3d_n = -1.;
        m_cl3d_id.clear();
        m_cl3d_pt.clear();
        m_cl3d_energy.clear();
        m_cl3d_eta.clear();
        m_cl3d_phi.clear();
        m_cl3d_clusters_n.clear();
        m_cl3d_clusters_id.clear();
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
        m_cl3d_genparticle_index.clear();

        // CALO-TRUTH TRIGGER CELLS INFORMATION
        m_tctruth_n = -1.;
        m_tctruth_id.clear();
        m_tctruth_subdet.clear();
        m_tctruth_zside.clear();
        m_tctruth_layer.clear();
        m_tctruth_waferu.clear();
        m_tctruth_waferv.clear();
        m_tctruth_wafertype.clear();
        m_tctruth_panel_number.clear();
        m_tctruth_panel_sector.clear();
        m_tctruth_cellu.clear();
        m_tctruth_cellv.clear();
        m_tctruth_data.clear();
        m_tctruth_uncompressedCharge.clear();
        m_tctruth_compressedCharge.clear();
        m_tctruth_pt.clear();
        m_tctruth_mipPt.clear();
        m_tctruth_energy.clear();
        m_tctruth_eta.clear();
        m_tctruth_phi.clear();
        m_tctruth_x.clear();
        m_tctruth_y.clear();
        m_tctruth_z.clear();
        m_tctruth_cluster_id.clear();
        m_tctruth_multicluster_id.clear();
        m_tctruth_multicluster_pt.clear();

        // CALO-TRUTH 3D CLUSTERS INFORMATION
        m_cl3dtruth_n = -1.;
        m_cl3dtruth_id.clear();
        m_cl3dtruth_pt.clear();
        m_cl3dtruth_energy.clear();
        m_cl3dtruth_eta.clear();
        m_cl3dtruth_phi.clear();
        m_cl3dtruth_clusters_n.clear();
        m_cl3dtruth_clusters_id.clear();
        m_cl3dtruth_showerlength.clear();
        m_cl3dtruth_coreshowerlength.clear();
        m_cl3dtruth_firstlayer.clear();
        m_cl3dtruth_maxlayer.clear();
        m_cl3dtruth_seetot.clear();
        m_cl3dtruth_seemax.clear();
        m_cl3dtruth_spptot.clear();
        m_cl3dtruth_sppmax.clear();
        m_cl3dtruth_szz.clear();
        m_cl3dtruth_srrtot.clear();
        m_cl3dtruth_srrmax.clear();
        m_cl3dtruth_srrmean.clear();
        m_cl3dtruth_emaxe.clear();
        m_cl3dtruth_hoe.clear();
        m_cl3dtruth_meanz.clear();
        m_cl3dtruth_layer10.clear();
        m_cl3dtruth_layer50.clear();
        m_cl3dtruth_layer90.clear();
        m_cl3dtruth_ntc67.clear();
        m_cl3dtruth_ntc90.clear();
        m_cl3dtruth_bdteg.clear();
        m_cl3dtruth_quality.clear();

        return 0;
    }

    int init() {
        // GENERAL INFORMATION
        m_skimT->Branch("run", &m_run); //, "run/F") ;
        m_skimT->Branch("event", &m_event); //, "event/F") ;
        m_skimT->Branch("lumi", &m_lumi); //, "lumi/F") ;

        // GENERATOR LEVEL INFORMATION
        m_skimT->Branch("genjet_n", &m_genjet_n); //, "genjet_n/F") ;
        m_skimT->Branch("genjet_pt", &m_genjet_pt); //, "genjet_pt/F") ;
        m_skimT->Branch("genjet_eta", &m_genjet_eta); //, "genjet_eta/F") ;
        m_skimT->Branch("genjet_phi", &m_genjet_phi); //, "genjet_phi/F") ;
        m_skimT->Branch("genjet_energy", &m_genjet_energy); //, "genjet_energy/F") ;
        m_skimT->Branch("genjet_mass", &m_genjet_mass); //, "genjet_mass/F") ;

        m_skimT->Branch("gentau_n", &m_gentau_n); //, "gentau_n/F") ;
        m_skimT->Branch("gentau_pt", &m_gentau_pt); //, "gentau_pt/F") ;
        m_skimT->Branch("gentau_eta", &m_gentau_eta); //, "gentau_eta/F") ;
        m_skimT->Branch("gentau_phi", &m_gentau_phi); //, "gentau_phi/F") ;
        m_skimT->Branch("gentau_energy", &m_gentau_energy); //, "gentau_energy/F") ;
        m_skimT->Branch("gentau_mass", &m_gentau_mass); //, "gentau_mass/F") ;

        m_skimT->Branch("gentau_vis_pt", &m_gentau_vis_pt); //, "gentau_vis_pt/F") ;
        m_skimT->Branch("gentau_vis_eta", &m_gentau_vis_eta); //, "gentau_vis_eta/F") ;
        m_skimT->Branch("gentau_vis_phi", &m_gentau_vis_phi); //, "gentau_vis_phi/F") ;
        m_skimT->Branch("gentau_vis_energy", &m_gentau_vis_energy); //, "gentau_vis_energy/F") ;
        m_skimT->Branch("gentau_vis_mass", &m_gentau_vis_mass); //, "gentau_vis_mass/F") ;
        m_skimT->Branch("gentau_decayMode", &m_gentau_decayMode); //, "gentau_decayMode/F") ;

        m_skimT->Branch("gentau_products_pt", &m_gentau_products_pt); //, "gentau_products_pt/F") ;
        m_skimT->Branch("gentau_products_eta", &m_gentau_products_eta); //, "gentau_products_eta/F") ;
        m_skimT->Branch("gentau_products_phi", &m_gentau_products_phi); //, "gentau_products_phi/F") ;
        m_skimT->Branch("gentau_products_energy", &m_gentau_products_energy); //, "gentau_products_energy/F") ;
        m_skimT->Branch("gentau_products_mass", &m_gentau_products_mass); //, "gentau_products_mass/F") ;
        m_skimT->Branch("gentau_products_id", &m_gentau_products_id); //, "gentau_products_id/F") ;
        m_skimT->Branch("gentau_totNproducts", &m_gentau_totNproducts); //, "gentau_totNproducts/F") ;
        m_skimT->Branch("gentau_totNgamma", &m_gentau_totNgamma); //, "gentau_totNgamma/F") ;
        m_skimT->Branch("gentau_totNpiZero", &m_gentau_totNpiZero); //, "gentau_totNpiZero/F") ;
        m_skimT->Branch("gentau_totNcharged", &m_gentau_totNcharged); //, "gentau_totNcharged/F") ;

        m_skimT->Branch("genpart_gen", &m_genpart_gen);

        // RECO TRIGGER CELLS INFORMATION
        m_skimT->Branch("tc_n", &m_tc_n); //, "tc_n/F") ;
        m_skimT->Branch("tc_id", &m_tc_id); //, "tc_id/F") ;
        m_skimT->Branch("tc_subdet", &m_tc_subdet); //, "tc_subdet/F") ;
        m_skimT->Branch("tc_zside", &m_tc_zside); //, "tc_zside/F") ;
        m_skimT->Branch("tc_layer", &m_tc_layer); //, "tc_layer/F") ;
        m_skimT->Branch("tc_waferu", &m_tc_waferu); //, "tc_waferu/F") ;
        m_skimT->Branch("tc_waferv", &m_tc_waferv); //, "tc_waferv/F") ;
        m_skimT->Branch("tc_wafertype", &m_tc_wafertype); //, "tc_wafertype/F") ;
        m_skimT->Branch("tc_panel_number", &m_tc_panel_number); //, "tc_panel_number/F") ;
        m_skimT->Branch("tc_panel_sector", &m_tc_panel_sector); //, "tc_panel_sector/F") ;
        m_skimT->Branch("tc_cellu", &m_tc_cellu); //, "tc_cellu/F") ;
        m_skimT->Branch("tc_cellv", &m_tc_cellv); //, "tc_cellv/F") ;
        m_skimT->Branch("tc_data", &m_tc_data); //, "tc_data/F") ;
        m_skimT->Branch("tc_uncompressedCharge", &m_tc_uncompressedCharge); //, "tc_uncompressedCharge/F") ;
        m_skimT->Branch("tc_compressedCharge", &m_tc_compressedCharge); //, "tc_compressedCharge/F") ;
        m_skimT->Branch("tc_pt", &m_tc_pt); //, "tc_pt/F") ;
        m_skimT->Branch("tc_mipPt", &m_tc_mipPt); //, "tc_mipPt/F") ;
        m_skimT->Branch("tc_energy", &m_tc_energy); //, "tc_energy/F") ;
        m_skimT->Branch("tc_eta", &m_tc_eta); //, "tc_eta/F") ;
        m_skimT->Branch("tc_phi", &m_tc_phi); //, "tc_phi/F") ;
        m_skimT->Branch("tc_x", &m_tc_x); //, "tc_x/F") ;
        m_skimT->Branch("tc_y", &m_tc_y); //, "tc_y/F") ;
        m_skimT->Branch("tc_z", &m_tc_z); //, "tc_z/F") ;
        m_skimT->Branch("tc_cluster_id", &m_tc_cluster_id); //, "tc_cluster_id/F") ;
        m_skimT->Branch("tc_multicluster_id", &m_tc_multicluster_id); //, "tc_multicluster_id/F") ;
        m_skimT->Branch("tc_multicluster_pt", &m_tc_multicluster_pt); //, "tc_multicluster_pt/F") ;
        m_skimT->Branch("tc_genparticle_index", &m_tc_genparticle_index);

        // RECO 3D CLUSTERS INFORMATION
        m_skimT->Branch("cl3d_n", &m_cl3d_n); //, "cl3d_n") ;
        m_skimT->Branch("cl3d_id", &m_cl3d_id); //, "cl3d_id") ;
        m_skimT->Branch("cl3d_pt", &m_cl3d_pt); //, "cl3d_pt") ;
        m_skimT->Branch("cl3d_energy", &m_cl3d_energy); //, "cl3d_energy") ;
        m_skimT->Branch("cl3d_eta", &m_cl3d_eta); //, "cl3d_eta") ;
        m_skimT->Branch("cl3d_phi", &m_cl3d_phi); //, "cl3d_phi") ;
        m_skimT->Branch("cl3d_clusters_n", &m_cl3d_clusters_n); //, "cl3d_clusters_n") ;
        m_skimT->Branch("cl3d_clusters_id", &m_cl3d_clusters_id); //, "cl3d_clusters_id") ;
        m_skimT->Branch("cl3d_showerlength", &m_cl3d_showerlength); //, "cl3d_showerlength/F") ;
        m_skimT->Branch("cl3d_coreshowerlength", &m_cl3d_coreshowerlength); //, "cl3d_coreshowerlength/F") ;
        m_skimT->Branch("cl3d_firstlayer", &m_cl3d_firstlayer); //, "cl3d_firstlayer/F") ;
        m_skimT->Branch("cl3d_maxlayer", &m_cl3d_maxlayer); //, "cl3d_maxlayer/F") ;
        m_skimT->Branch("cl3d_seetot", &m_cl3d_seetot); //, "cl3d_seetot/F") ;
        m_skimT->Branch("cl3d_seemax", &m_cl3d_seemax); //, "cl3d_seemax/F") ;
        m_skimT->Branch("cl3d_spptot", &m_cl3d_spptot); //, "cl3d_spptot/F") ;
        m_skimT->Branch("cl3d_sppmax", &m_cl3d_sppmax); //, "cl3d_sppmax/F") ;
        m_skimT->Branch("cl3d_szz", &m_cl3d_szz); //, "cl3d_szz/F") ;
        m_skimT->Branch("cl3d_srrtot", &m_cl3d_srrtot); //, "cl3d_srrtot/F") ;
        m_skimT->Branch("cl3d_srrmax", &m_cl3d_srrmax); //, "cl3d_srrmax/F") ;
        m_skimT->Branch("cl3d_srrmean", &m_cl3d_srrmean); //, "cl3d_srrmean/F") ;
        m_skimT->Branch("cl3d_emaxe", &m_cl3d_emaxe); //, "cl3d_emaxe/F") ;
        m_skimT->Branch("cl3d_hoe", &m_cl3d_hoe); //, "cl3d_hoe/F") ;
        m_skimT->Branch("cl3d_meanz", &m_cl3d_meanz); //, "cl3d_meanz/F") ;
        m_skimT->Branch("cl3d_layer10", &m_cl3d_layer10); //, "cl3d_layer10/F") ;
        m_skimT->Branch("cl3d_layer50", &m_cl3d_layer50); //, "cl3d_layer50/F") ;
        m_skimT->Branch("cl3d_layer90", &m_cl3d_layer90); //, "cl3d_layer90/F") ;
        m_skimT->Branch("cl3d_ntc67", &m_cl3d_ntc67); //, "cl3d_ntc67/F") ;
        m_skimT->Branch("cl3d_ntc90", &m_cl3d_ntc90); //, "cl3d_ntc90/F") ;
        m_skimT->Branch("cl3d_bdteg", &m_cl3d_bdteg); //, "cl3d_bdteg/F") ;
        m_skimT->Branch("cl3d_quality", &m_cl3d_quality); //, "cl3d_quality/F") ;
        m_skimT->Branch("cl3d_genparticle_index", &m_cl3d_genparticle_index);

        // CALO-THRUTH TRIGGER CELLS INFORMATION
        m_skimT->Branch("tctruth_n", &m_tctruth_n); //, "tctruth_n/F") ;
        m_skimT->Branch("tctruth_id", &m_tctruth_id); //, "tctruth_id/F") ;
        m_skimT->Branch("tctruth_subdet", &m_tctruth_subdet); //, "tctruth_subdet/F") ;
        m_skimT->Branch("tctruth_zside", &m_tctruth_zside); //, "tctruth_zside/F") ;
        m_skimT->Branch("tctruth_layer", &m_tctruth_layer); //, "tctruth_layer/F") ;
        m_skimT->Branch("tctruth_waferu", &m_tctruth_waferu); //, "tctruth_waferu/F") ;
        m_skimT->Branch("tctruth_waferv", &m_tctruth_waferv); //, "tctruth_waferv/F") ;
        m_skimT->Branch("tctruth_wafertype", &m_tctruth_wafertype); //, "tctruth_wafertype/F") ;
        m_skimT->Branch("tctruth_panel_number", &m_tctruth_panel_number); //, "tctruth_panel_number/F") ;
        m_skimT->Branch("tctruth_panel_sector", &m_tctruth_panel_sector); //, "tctruth_panel_sector/F") ;
        m_skimT->Branch("tctruth_cellu", &m_tctruth_cellu); //, "tctruth_cellu/F") ;
        m_skimT->Branch("tctruth_cellv", &m_tctruth_cellv); //, "tctruth_cellv/F") ;
        m_skimT->Branch("tctruth_data", &m_tctruth_data); //, "tctruth_data/F") ;
        m_skimT->Branch("tctruth_uncompressedCharge", &m_tctruth_uncompressedCharge); //, "tctruth_uncompressedCharge/F") ;
        m_skimT->Branch("tctruth_compressedCharge", &m_tctruth_compressedCharge); //, "tctruth_compressedCharge/F") ;
        m_skimT->Branch("tctruth_pt", &m_tctruth_pt); //, "tctruth_pt/F") ;
        m_skimT->Branch("tctruth_mipPt", &m_tctruth_mipPt); //, "tctruth_mipPt/F") ;
        m_skimT->Branch("tctruth_energy", &m_tctruth_energy); //, "tctruth_energy/F") ;
        m_skimT->Branch("tctruth_eta", &m_tctruth_eta); //, "tctruth_eta/F") ;
        m_skimT->Branch("tctruth_phi", &m_tctruth_phi); //, "tctruth_phi/F") ;
        m_skimT->Branch("tctruth_x", &m_tctruth_x); //, "tctruth_x/F") ;
        m_skimT->Branch("tctruth_y", &m_tctruth_y); //, "tctruth_y/F") ;
        m_skimT->Branch("tctruth_z", &m_tctruth_z); //, "tctruth_z/F") ;
        m_skimT->Branch("tctruth_cluster_id", &m_tctruth_cluster_id); //, "tctruth_cluster_id/F") ;
        m_skimT->Branch("tctruth_multicluster_id", &m_tctruth_multicluster_id); //, "tctruth_multicluster_id/F") ;
        m_skimT->Branch("tctruth_multicluster_pt", &m_tctruth_multicluster_pt); //, "tctruth_multicluster_pt/F") ;

        // CALO-TRUTH 3D CLUSTERS INFORMATION
        m_skimT->Branch("cl3dtruth_n", &m_cl3dtruth_n); //, "cl3dtruth_n") ;
        m_skimT->Branch("cl3dtruth_id", &m_cl3dtruth_id); //, "cl3dtruth_id") ;
        m_skimT->Branch("cl3dtruth_pt", &m_cl3dtruth_pt); //, "cl3dtruth_pt") ;
        m_skimT->Branch("cl3dtruth_energy", &m_cl3dtruth_energy); //, "cl3dtruth_energy") ;
        m_skimT->Branch("cl3dtruth_eta", &m_cl3dtruth_eta); //, "cl3dtruth_eta") ;
        m_skimT->Branch("cl3dtruth_phi", &m_cl3dtruth_phi); //, "cl3dtruth_phi") ;
        m_skimT->Branch("cl3dtruth_clusters_n", &m_cl3dtruth_clusters_n); //, "cl3dtruth_clusters_n") ;
        m_skimT->Branch("cl3dtruth_clusters_id", &m_cl3dtruth_clusters_id); //, "cl3dtruth_clusters_id") ;
        m_skimT->Branch("cl3dtruth_showerlength", &m_cl3dtruth_showerlength); //, "cl3dtruth_showerlength/F") ;
        m_skimT->Branch("cl3dtruth_coreshowerlength", &m_cl3dtruth_coreshowerlength); //, "cl3dtruth_coreshowerlength/F") ;
        m_skimT->Branch("cl3dtruth_firstlayer", &m_cl3dtruth_firstlayer); //, "cl3dtruth_firstlayer/F") ;
        m_skimT->Branch("cl3dtruth_maxlayer", &m_cl3dtruth_maxlayer); //, "cl3dtruth_maxlayer/F") ;
        m_skimT->Branch("cl3dtruth_seetot", &m_cl3dtruth_seetot); //, "cl3dtruth_seetot/F") ;
        m_skimT->Branch("cl3dtruth_seemax", &m_cl3dtruth_seemax); //, "cl3dtruth_seemax/F") ;
        m_skimT->Branch("cl3dtruth_spptot", &m_cl3dtruth_spptot); //, "cl3dtruth_spptot/F") ;
        m_skimT->Branch("cl3dtruth_sppmax", &m_cl3dtruth_sppmax); //, "cl3dtruth_sppmax/F") ;
        m_skimT->Branch("cl3dtruth_szz", &m_cl3dtruth_szz); //, "cl3dtruth_szz/F") ;
        m_skimT->Branch("cl3dtruth_srrtot", &m_cl3dtruth_srrtot); //, "cl3dtruth_srrtot/F") ;
        m_skimT->Branch("cl3dtruth_srrmax", &m_cl3dtruth_srrmax); //, "cl3dtruth_srrmax/F") ;
        m_skimT->Branch("cl3dtruth_srrmean", &m_cl3dtruth_srrmean); //, "cl3dtruth_srrmean/F") ;
        m_skimT->Branch("cl3dtruth_emaxe", &m_cl3dtruth_emaxe); //, "cl3dtruth_emaxe/F") ;
        m_skimT->Branch("cl3dtruth_hoe", &m_cl3dtruth_hoe); //, "cl3dtruth_hoe/F") ;
        m_skimT->Branch("cl3dtruth_meanz", &m_cl3dtruth_meanz); //, "cl3dtruth_meanz/F") ;
        m_skimT->Branch("cl3dtruth_layer10", &m_cl3dtruth_layer10); //, "cl3dtruth_layer10/F") ;
        m_skimT->Branch("cl3dtruth_layer50", &m_cl3dtruth_layer50); //, "cl3dtruth_layer50/F") ;
        m_skimT->Branch("cl3dtruth_layer90", &m_cl3dtruth_layer90); //, "cl3dtruth_layer90/F") ;
        m_skimT->Branch("cl3dtruth_ntc67", &m_cl3dtruth_ntc67); //, "cl3dtruth_ntc67/F") ;
        m_skimT->Branch("cl3dtruth_ntc90", &m_cl3dtruth_ntc90); //, "cl3dtruth_ntc90/F") ;
        m_skimT->Branch("cl3dtruth_bdteg", &m_cl3dtruth_bdteg); //, "cl3dtruth_bdteg/F") ;
        m_skimT->Branch("cl3dtruth_quality", &m_cl3dtruth_quality); //, "cl3dtruth_quality/F") ;

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

    vector<int>    m_genpart_gen;

    // RECO TRIGGER CELLS INFORMATION
    int                     m_tc_n;
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
    vector<int>             m_tc_genparticle_index;

    // RECO 3D CLUSTERS INFORMATION
    int                             m_cl3d_n;
    vector<unsigned int>            m_cl3d_id;
    vector<float>                   m_cl3d_pt;
    vector<float>                   m_cl3d_energy;
    vector<float>                   m_cl3d_eta;
    vector<float>                   m_cl3d_phi;
    vector<int>                     m_cl3d_clusters_n;
    vector<vector<unsigned int>>    m_cl3d_clusters_id;
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
    vector<int>                     m_cl3d_genparticle_index;

    // CALO-TRUTH TRIGGER CELLS INFORMATION
    int                     m_tctruth_n;
    vector<unsigned int>    m_tctruth_id;
    vector<int>             m_tctruth_subdet;
    vector<int>             m_tctruth_zside;
    vector<int>             m_tctruth_layer;
    vector<int>             m_tctruth_waferu;
    vector<int>             m_tctruth_waferv;
    vector<int>             m_tctruth_wafertype;
    vector<int>             m_tctruth_panel_number;
    vector<int>             m_tctruth_panel_sector;
    vector<int>             m_tctruth_cellu;
    vector<int>             m_tctruth_cellv;  
    vector<unsigned int>    m_tctruth_data;
    vector<unsigned int>    m_tctruth_uncompressedCharge;
    vector<unsigned int>    m_tctruth_compressedCharge;
    vector<float>           m_tctruth_pt;
    vector<float>           m_tctruth_mipPt;
    vector<float>           m_tctruth_energy;
    vector<float>           m_tctruth_eta;
    vector<float>           m_tctruth_phi;
    vector<float>           m_tctruth_x;
    vector<float>           m_tctruth_y;
    vector<float>           m_tctruth_z;
    vector<unsigned int>    m_tctruth_cluster_id;
    vector<unsigned int>    m_tctruth_multicluster_id;
    vector<float>           m_tctruth_multicluster_pt;

    // CALO-TRUTH 3D CLUSTERS INFORMATION
    int                             m_cl3dtruth_n;
    vector<unsigned int>            m_cl3dtruth_id;
    vector<float>                   m_cl3dtruth_pt;
    vector<float>                   m_cl3dtruth_energy;
    vector<float>                   m_cl3dtruth_eta;
    vector<float>                   m_cl3dtruth_phi;
    vector<int>                     m_cl3dtruth_clusters_n;
    vector<vector<unsigned int>>    m_cl3dtruth_clusters_id;
    vector<int>                     m_cl3dtruth_showerlength;
    vector<int>                     m_cl3dtruth_coreshowerlength;
    vector<int>                     m_cl3dtruth_firstlayer;
    vector<int>                     m_cl3dtruth_maxlayer;
    vector<float>                   m_cl3dtruth_seetot;
    vector<float>                   m_cl3dtruth_seemax;
    vector<float>                   m_cl3dtruth_spptot;
    vector<float>                   m_cl3dtruth_sppmax;
    vector<float>                   m_cl3dtruth_szz;
    vector<float>                   m_cl3dtruth_srrtot;
    vector<float>                   m_cl3dtruth_srrmax;
    vector<float>                   m_cl3dtruth_srrmean;
    vector<float>                   m_cl3dtruth_emaxe;
    vector<float>                   m_cl3dtruth_hoe;
    vector<float>                   m_cl3dtruth_meanz;
    vector<float>                   m_cl3dtruth_layer10;
    vector<float>                   m_cl3dtruth_layer50;
    vector<float>                   m_cl3dtruth_layer90;
    vector<float>                   m_cl3dtruth_ntc67;
    vector<float>                   m_cl3dtruth_ntc90;
    vector<float>                   m_cl3dtruth_bdteg;
    vector<int>                     m_cl3dtruth_quality;
};

#endif