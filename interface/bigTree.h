//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Apr 18 12:11:17 2021 by ROOT version 6.20/06
// from TTree HGCalTriggerNtuple/HGCalTriggerNtuple
// found on file: TauTest.root
//////////////////////////////////////////////////////////

#ifndef bigTree_h
#define bigTree_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include <vector>

class bigTree {
public :
   TChain          *fChain;   //!pointer to the analyzed TTree or TChain

   // Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           run;
   Int_t           event;
   Int_t           lumi;
   Int_t           gen_n;
   Int_t           gen_PUNumInt;
   Float_t         gen_TrueNumInt;
   Float_t         vtx_x;
   Float_t         vtx_y;
   Float_t         vtx_z;
   std::vector<float>   *gen_eta;
   std::vector<float>   *gen_phi;
   std::vector<float>   *gen_pt;
   std::vector<float>   *gen_energy;
   std::vector<int>     *gen_charge;
   std::vector<int>     *gen_pdgid;
   std::vector<int>     *gen_status;
   std::vector<std::vector<int> > *gen_daughters;
   std::vector<float>   *genpart_eta;
   std::vector<float>   *genpart_phi;
   std::vector<float>   *genpart_pt;
   std::vector<float>   *genpart_energy;
   std::vector<float>   *genpart_dvx;
   std::vector<float>   *genpart_dvy;
   std::vector<float>   *genpart_dvz;
   std::vector<float>   *genpart_ovx;
   std::vector<float>   *genpart_ovy;
   std::vector<float>   *genpart_ovz;
   std::vector<int>     *genpart_mother;
   std::vector<float>   *genpart_exphi;
   std::vector<float>   *genpart_exeta;
   std::vector<float>   *genpart_exx;
   std::vector<float>   *genpart_exy;
   std::vector<float>   *genpart_fbrem;
   std::vector<int>     *genpart_pid;
   std::vector<int>     *genpart_gen;
   std::vector<int>     *genpart_reachedEE;
   std::vector<bool>    *genpart_fromBeamPipe;
   std::vector<std::vector<float> > *genpart_posx;
   std::vector<std::vector<float> > *genpart_posy;
   std::vector<std::vector<float> > *genpart_posz;
   Int_t           genjet_n;
   std::vector<float>   *genjet_energy;
   std::vector<float>   *genjet_pt;
   std::vector<float>   *genjet_eta;
   std::vector<float>   *genjet_phi;
   std::vector<float>   *gentau_pt;
   std::vector<float>   *gentau_eta;
   std::vector<float>   *gentau_phi;
   std::vector<float>   *gentau_energy;
   std::vector<float>   *gentau_mass;
   std::vector<float>   *gentau_vis_pt;
   std::vector<float>   *gentau_vis_eta;
   std::vector<float>   *gentau_vis_phi;
   std::vector<float>   *gentau_vis_energy;
   std::vector<float>   *gentau_vis_mass;
   std::vector<std::vector<float> > *gentau_products_pt;
   std::vector<std::vector<float> > *gentau_products_eta;
   std::vector<std::vector<float> > *gentau_products_phi;
   std::vector<std::vector<float> > *gentau_products_energy;
   std::vector<std::vector<float> > *gentau_products_mass;
   std::vector<std::vector<int> > *gentau_products_id;
   std::vector<int>     *gentau_decayMode;
   std::vector<int>     *gentau_totNproducts;
   std::vector<int>     *gentau_totNgamma;
   std::vector<int>     *gentau_totNpiZero;
   std::vector<int>     *gentau_totNcharged;
   Int_t           hgcdigi_n;
   std::vector<int>     *hgcdigi_id;
   std::vector<int>     *hgcdigi_subdet;
   std::vector<int>     *hgcdigi_zside;
   std::vector<int>     *hgcdigi_layer;
   std::vector<int>     *hgcdigi_wafertype;
   std::vector<float>   *hgcdigi_eta;
   std::vector<float>   *hgcdigi_phi;
   std::vector<float>   *hgcdigi_z;
   std::vector<unsigned int> *hgcdigi_data_BX2;
   std::vector<int>     *hgcdigi_isadc_BX2;
   std::vector<int>     *hgcdigi_waferu;
   std::vector<int>     *hgcdigi_waferv;
   std::vector<int>     *hgcdigi_cellu;
   std::vector<int>     *hgcdigi_cellv;
   std::vector<int>     *hgcdigi_wafer;
   std::vector<int>     *hgcdigi_cell;
   Int_t           bhdigi_n;
   std::vector<int>     *bhdigi_id;
   std::vector<int>     *bhdigi_subdet;
   std::vector<int>     *bhdigi_zside;
   std::vector<int>     *bhdigi_layer;
   std::vector<int>     *bhdigi_ieta;
   std::vector<int>     *bhdigi_iphi;
   std::vector<float>   *bhdigi_eta;
   std::vector<float>   *bhdigi_phi;
   std::vector<float>   *bhdigi_z;
   std::vector<unsigned int> *bhdigi_data_BX2;
   std::vector<int>     *bhdigi_isadc_BX2;
   Int_t           tc_n;
   std::vector<unsigned int> *tc_id;
   std::vector<int>     *tc_subdet;
   std::vector<int>     *tc_zside;
   std::vector<int>     *tc_layer;
   std::vector<int>     *tc_wafer;
   std::vector<int>     *tc_waferu;
   std::vector<int>     *tc_waferv;
   std::vector<int>     *tc_wafertype;
   std::vector<int>     *tc_panel_number;
   std::vector<int>     *tc_panel_sector;
   std::vector<int>     *tc_cell;
   std::vector<int>     *tc_cellu;
   std::vector<int>     *tc_cellv;
   std::vector<unsigned int> *tc_data;
   std::vector<unsigned int> *tc_uncompressedCharge;
   std::vector<unsigned int> *tc_compressedCharge;
   std::vector<float>   *tc_pt;
   std::vector<float>   *tc_mipPt;
   std::vector<float>   *tc_energy;
   std::vector<float>   *tc_eta;
   std::vector<float>   *tc_phi;
   std::vector<float>   *tc_x;
   std::vector<float>   *tc_y;
   std::vector<float>   *tc_z;
   std::vector<unsigned int> *tc_cluster_id;
   std::vector<unsigned int> *tc_multicluster_id;
   std::vector<float>   *tc_multicluster_pt;
   std::vector<int>     *tc_genparticle_index;
   Int_t           ts_n;
   std::vector<unsigned int> *ts_id;
   std::vector<int>     *ts_subdet;
   std::vector<int>     *ts_zside;
   std::vector<int>     *ts_layer;
   std::vector<int>     *ts_wafer;
   std::vector<int>     *ts_wafertype;
   std::vector<int>     *ts_panel_number;
   std::vector<int>     *ts_panel_sector;
   std::vector<unsigned int> *ts_data;
   std::vector<float>   *ts_pt;
   std::vector<float>   *ts_mipPt;
   std::vector<float>   *ts_energy;
   std::vector<float>   *ts_eta;
   std::vector<float>   *ts_phi;
   std::vector<float>   *ts_x;
   std::vector<float>   *ts_y;
   std::vector<float>   *ts_z;
   Int_t           cl3d_n;
   std::vector<unsigned int> *cl3d_id;
   std::vector<float>   *cl3d_pt;
   std::vector<float>   *cl3d_energy;
   std::vector<float>   *cl3d_eta;
   std::vector<float>   *cl3d_phi;
   std::vector<int>     *cl3d_clusters_n;
   std::vector<std::vector<unsigned int> > *cl3d_clusters_id;
   std::vector<int>     *cl3d_showerlength;
   std::vector<int>     *cl3d_coreshowerlength;
   std::vector<int>     *cl3d_firstlayer;
   std::vector<int>     *cl3d_maxlayer;
   std::vector<float>   *cl3d_seetot;
   std::vector<float>   *cl3d_seemax;
   std::vector<float>   *cl3d_spptot;
   std::vector<float>   *cl3d_sppmax;
   std::vector<float>   *cl3d_szz;
   std::vector<float>   *cl3d_srrtot;
   std::vector<float>   *cl3d_srrmax;
   std::vector<float>   *cl3d_srrmean;
   std::vector<float>   *cl3d_emaxe;
   std::vector<float>   *cl3d_hoe;
   std::vector<float>   *cl3d_meanz;
   std::vector<float>   *cl3d_layer10;
   std::vector<float>   *cl3d_layer50;
   std::vector<float>   *cl3d_layer90;
   std::vector<float>   *cl3d_ntc67;
   std::vector<float>   *cl3d_ntc90;
   std::vector<float>   *cl3d_bdteg;
   std::vector<int>     *cl3d_quality;
   std::vector<std::vector<float> > *cl3d_ipt;
   std::vector<std::vector<float> > *cl3d_ienergy;
   Int_t           tower_n;
   std::vector<float>   *tower_pt;
   std::vector<float>   *tower_energy;
   std::vector<float>   *tower_eta;
   std::vector<float>   *tower_phi;
   std::vector<float>   *tower_etEm;
   std::vector<float>   *tower_etHad;
   std::vector<int>     *tower_iEta;
   std::vector<int>     *tower_iPhi;
   Int_t           cl3dfulltruth_n;
   std::vector<unsigned int> *cl3dfulltruth_id;
   std::vector<float>   *cl3dfulltruth_pt;
   std::vector<float>   *cl3dfulltruth_energy;
   std::vector<float>   *cl3dfulltruth_eta;
   std::vector<float>   *cl3dfulltruth_phi;
   std::vector<int>     *cl3dfulltruth_clusters_n;
   std::vector<std::vector<unsigned int> > *cl3dfulltruth_clusters_id;
   std::vector<int>     *cl3dfulltruth_showerlength;
   std::vector<int>     *cl3dfulltruth_coreshowerlength;
   std::vector<int>     *cl3dfulltruth_firstlayer;
   std::vector<int>     *cl3dfulltruth_maxlayer;
   std::vector<float>   *cl3dfulltruth_seetot;
   std::vector<float>   *cl3dfulltruth_seemax;
   std::vector<float>   *cl3dfulltruth_spptot;
   std::vector<float>   *cl3dfulltruth_sppmax;
   std::vector<float>   *cl3dfulltruth_szz;
   std::vector<float>   *cl3dfulltruth_srrtot;
   std::vector<float>   *cl3dfulltruth_srrmax;
   std::vector<float>   *cl3dfulltruth_srrmean;
   std::vector<float>   *cl3dfulltruth_emaxe;
   std::vector<float>   *cl3dfulltruth_hoe;
   std::vector<float>   *cl3dfulltruth_meanz;
   std::vector<float>   *cl3dfulltruth_layer10;
   std::vector<float>   *cl3dfulltruth_layer50;
   std::vector<float>   *cl3dfulltruth_layer90;
   std::vector<float>   *cl3dfulltruth_ntc67;
   std::vector<float>   *cl3dfulltruth_ntc90;
   std::vector<float>   *cl3dfulltruth_bdteg;
   std::vector<int>     *cl3dfulltruth_quality;
   std::vector<std::vector<float> > *cl3dfulltruth_ipt;
   std::vector<std::vector<float> > *cl3dfulltruth_ienergy;
   Int_t           tctruth_n;
   std::vector<unsigned int> *tctruth_id;
   std::vector<int>     *tctruth_subdet;
   std::vector<int>     *tctruth_zside;
   std::vector<int>     *tctruth_layer;
   std::vector<int>     *tctruth_wafer;
   std::vector<int>     *tctruth_waferu;
   std::vector<int>     *tctruth_waferv;
   std::vector<int>     *tctruth_wafertype;
   std::vector<int>     *tctruth_panel_number;
   std::vector<int>     *tctruth_panel_sector;
   std::vector<int>     *tctruth_cell;
   std::vector<int>     *tctruth_cellu;
   std::vector<int>     *tctruth_cellv;
   std::vector<unsigned int> *tctruth_data;
   std::vector<unsigned int> *tctruth_uncompressedCharge;
   std::vector<unsigned int> *tctruth_compressedCharge;
   std::vector<float>   *tctruth_pt;
   std::vector<float>   *tctruth_mipPt;
   std::vector<float>   *tctruth_energy;
   std::vector<float>   *tctruth_eta;
   std::vector<float>   *tctruth_phi;
   std::vector<float>   *tctruth_x;
   std::vector<float>   *tctruth_y;
   std::vector<float>   *tctruth_z;
   std::vector<unsigned int> *tctruth_cluster_id;
   std::vector<unsigned int> *tctruth_multicluster_id;
   std::vector<float>   *tctruth_multicluster_pt;
   Int_t           cl3dtruth_n;
   std::vector<unsigned int> *cl3dtruth_id;
   std::vector<float>   *cl3dtruth_pt;
   std::vector<float>   *cl3dtruth_energy;
   std::vector<float>   *cl3dtruth_eta;
   std::vector<float>   *cl3dtruth_phi;
   std::vector<int>     *cl3dtruth_clusters_n;
   std::vector<std::vector<unsigned int> > *cl3dtruth_clusters_id;
   std::vector<int>     *cl3dtruth_showerlength;
   std::vector<int>     *cl3dtruth_coreshowerlength;
   std::vector<int>     *cl3dtruth_firstlayer;
   std::vector<int>     *cl3dtruth_maxlayer;
   std::vector<float>   *cl3dtruth_seetot;
   std::vector<float>   *cl3dtruth_seemax;
   std::vector<float>   *cl3dtruth_spptot;
   std::vector<float>   *cl3dtruth_sppmax;
   std::vector<float>   *cl3dtruth_szz;
   std::vector<float>   *cl3dtruth_srrtot;
   std::vector<float>   *cl3dtruth_srrmax;
   std::vector<float>   *cl3dtruth_srrmean;
   std::vector<float>   *cl3dtruth_emaxe;
   std::vector<float>   *cl3dtruth_hoe;
   std::vector<float>   *cl3dtruth_meanz;
   std::vector<float>   *cl3dtruth_layer10;
   std::vector<float>   *cl3dtruth_layer50;
   std::vector<float>   *cl3dtruth_layer90;
   std::vector<float>   *cl3dtruth_ntc67;
   std::vector<float>   *cl3dtruth_ntc90;
   std::vector<float>   *cl3dtruth_bdteg;
   std::vector<int>     *cl3dtruth_quality;
   std::vector<std::vector<float> > *cl3dtruth_ipt;
   std::vector<std::vector<float> > *cl3dtruth_ienergy;
   Int_t           towertruth_n;
   std::vector<float>   *towertruth_pt;
   std::vector<float>   *towertruth_energy;
   std::vector<float>   *towertruth_eta;
   std::vector<float>   *towertruth_phi;
   std::vector<float>   *towertruth_etEm;
   std::vector<float>   *towertruth_etHad;
   std::vector<int>     *towertruth_iEta;
   std::vector<int>     *towertruth_iPhi;

   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_event;   //!
   TBranch        *b_lumi;   //!
   TBranch        *b_gen_n;   //!
   TBranch        *b_gen_PUNumInt;   //!
   TBranch        *b_gen_TrueNumInt;   //!
   TBranch        *b_vtx_x;   //!
   TBranch        *b_vtx_y;   //!
   TBranch        *b_vtx_z;   //!
   TBranch        *b_gen_eta;   //!
   TBranch        *b_gen_phi;   //!
   TBranch        *b_gen_pt;   //!
   TBranch        *b_gen_energy;   //!
   TBranch        *b_gen_charge;   //!
   TBranch        *b_gen_pdgid;   //!
   TBranch        *b_gen_status;   //!
   TBranch        *b_gen_daughters;   //!
   TBranch        *b_genpart_eta;   //!
   TBranch        *b_genpart_phi;   //!
   TBranch        *b_genpart_pt;   //!
   TBranch        *b_genpart_energy;   //!
   TBranch        *b_genpart_dvx;   //!
   TBranch        *b_genpart_dvy;   //!
   TBranch        *b_genpart_dvz;   //!
   TBranch        *b_genpart_ovx;   //!
   TBranch        *b_genpart_ovy;   //!
   TBranch        *b_genpart_ovz;   //!
   TBranch        *b_genpart_mother;   //!
   TBranch        *b_genpart_exphi;   //!
   TBranch        *b_genpart_exeta;   //!
   TBranch        *b_genpart_exx;   //!
   TBranch        *b_genpart_exy;   //!
   TBranch        *b_genpart_fbrem;   //!
   TBranch        *b_genpart_pid;   //!
   TBranch        *b_genpart_gen;   //!
   TBranch        *b_genpart_reachedEE;   //!
   TBranch        *b_genpart_fromBeamPipe;   //!
   TBranch        *b_genpart_posx;   //!
   TBranch        *b_genpart_posy;   //!
   TBranch        *b_genpart_posz;   //!
   TBranch        *b_genjet_n;   //!
   TBranch        *b_genjet_energy;   //!
   TBranch        *b_genjet_pt;   //!
   TBranch        *b_genjet_eta;   //!
   TBranch        *b_genjet_phi;   //!
   TBranch        *b_gentau_pt;   //!
   TBranch        *b_gentau_eta;   //!
   TBranch        *b_gentau_phi;   //!
   TBranch        *b_gentau_energy;   //!
   TBranch        *b_gentau_mass;   //!
   TBranch        *b_gentau_vis_pt;   //!
   TBranch        *b_gentau_vis_eta;   //!
   TBranch        *b_gentau_vis_phi;   //!
   TBranch        *b_gentau_vis_energy;   //!
   TBranch        *b_gentau_vis_mass;   //!
   TBranch        *b_gentau_products_pt;   //!
   TBranch        *b_gentau_products_eta;   //!
   TBranch        *b_gentau_products_phi;   //!
   TBranch        *b_gentau_products_energy;   //!
   TBranch        *b_gentau_products_mass;   //!
   TBranch        *b_gentau_products_id;   //!
   TBranch        *b_gentau_decayMode;   //!
   TBranch        *b_gentau_totNproducts;   //!
   TBranch        *b_gentau_totNgamma;   //!
   TBranch        *b_gentau_totNpiZero;   //!
   TBranch        *b_gentau_totNcharged;   //!
   TBranch        *b_hgcdigi_n;   //!
   TBranch        *b_hgcdigi_id;   //!
   TBranch        *b_hgcdigi_subdet;   //!
   TBranch        *b_hgcdigi_zside;   //!
   TBranch        *b_hgcdigi_layer;   //!
   TBranch        *b_hgcdigi_wafertype;   //!
   TBranch        *b_hgcdigi_eta;   //!
   TBranch        *b_hgcdigi_phi;   //!
   TBranch        *b_hgcdigi_z;   //!
   TBranch        *b_hgcdigi_data_BX2;   //!
   TBranch        *b_hgcdigi_isadc_BX2;   //!
   TBranch        *b_hgcdigi_waferu;   //!
   TBranch        *b_hgcdigi_waferv;   //!
   TBranch        *b_hgcdigi_cellu;   //!
   TBranch        *b_hgcdigi_cellv;   //!
   TBranch        *b_hgcdigi_wafer;   //!
   TBranch        *b_hgcdigi_cell;   //!
   TBranch        *b_bhdigi_n;   //!
   TBranch        *b_bhdigi_id;   //!
   TBranch        *b_bhdigi_subdet;   //!
   TBranch        *b_bhdigi_zside;   //!
   TBranch        *b_bhdigi_layer;   //!
   TBranch        *b_bhdigi_ieta;   //!
   TBranch        *b_bhdigi_iphi;   //!
   TBranch        *b_bhdigi_eta;   //!
   TBranch        *b_bhdigi_phi;   //!
   TBranch        *b_bhdigi_z;   //!
   TBranch        *b_bhdigi_data_BX2;   //!
   TBranch        *b_bhdigi_isadc_BX2;   //!
   TBranch        *b_tc_n;   //!
   TBranch        *b_tc_id;   //!
   TBranch        *b_tc_subdet;   //!
   TBranch        *b_tc_zside;   //!
   TBranch        *b_tc_layer;   //!
   TBranch        *b_tc_wafer;   //!
   TBranch        *b_tc_waferu;   //!
   TBranch        *b_tc_waferv;   //!
   TBranch        *b_tc_wafertype;   //!
   TBranch        *b_tc_panel_number;   //!
   TBranch        *b_tc_panel_sector;   //!
   TBranch        *b_tc_cell;   //!
   TBranch        *b_tc_cellu;   //!
   TBranch        *b_tc_cellv;   //!
   TBranch        *b_tc_data;   //!
   TBranch        *b_tc_uncompressedCharge;   //!
   TBranch        *b_tc_compressedCharge;   //!
   TBranch        *b_tc_pt;   //!
   TBranch        *b_tc_mipPt;   //!
   TBranch        *b_tc_energy;   //!
   TBranch        *b_tc_eta;   //!
   TBranch        *b_tc_phi;   //!
   TBranch        *b_tc_x;   //!
   TBranch        *b_tc_y;   //!
   TBranch        *b_tc_z;   //!
   TBranch        *b_tc_cluster_id;   //!
   TBranch        *b_tc_multicluster_id;   //!
   TBranch        *b_tc_multicluster_pt;   //!
   TBranch        *b_tc_genparticle_index;   //!
   TBranch        *b_ts_n;   //!
   TBranch        *b_ts_id;   //!
   TBranch        *b_ts_subdet;   //!
   TBranch        *b_ts_zside;   //!
   TBranch        *b_ts_layer;   //!
   TBranch        *b_ts_wafer;   //!
   TBranch        *b_ts_wafertype;   //!
   TBranch        *b_ts_panel_number;   //!
   TBranch        *b_ts_panel_sector;   //!
   TBranch        *b_ts_data;   //!
   TBranch        *b_ts_pt;   //!
   TBranch        *b_ts_mipPt;   //!
   TBranch        *b_ts_energy;   //!
   TBranch        *b_ts_eta;   //!
   TBranch        *b_ts_phi;   //!
   TBranch        *b_ts_x;   //!
   TBranch        *b_ts_y;   //!
   TBranch        *b_ts_z;   //!
   TBranch        *b_cl3d_n;   //!
   TBranch        *b_cl3d_id;   //!
   TBranch        *b_cl3d_pt;   //!
   TBranch        *b_cl3d_energy;   //!
   TBranch        *b_cl3d_eta;   //!
   TBranch        *b_cl3d_phi;   //!
   TBranch        *b_cl3d_clusters_n;   //!
   TBranch        *b_cl3d_clusters_id;   //!
   TBranch        *b_cl3d_showerlength;   //!
   TBranch        *b_cl3d_coreshowerlength;   //!
   TBranch        *b_cl3d_firstlayer;   //!
   TBranch        *b_cl3d_maxlayer;   //!
   TBranch        *b_cl3d_seetot;   //!
   TBranch        *b_cl3d_seemax;   //!
   TBranch        *b_cl3d_spptot;   //!
   TBranch        *b_cl3d_sppmax;   //!
   TBranch        *b_cl3d_szz;   //!
   TBranch        *b_cl3d_srrtot;   //!
   TBranch        *b_cl3d_srrmax;   //!
   TBranch        *b_cl3d_srrmean;   //!
   TBranch        *b_cl3d_emaxe;   //!
   TBranch        *b_cl3d_hoe;   //!
   TBranch        *b_cl3d_meanz;   //!
   TBranch        *b_cl3d_layer10;   //!
   TBranch        *b_cl3d_layer50;   //!
   TBranch        *b_cl3d_layer90;   //!
   TBranch        *b_cl3d_ntc67;   //!
   TBranch        *b_cl3d_ntc90;   //!
   TBranch        *b_cl3d_bdteg;   //!
   TBranch        *b_cl3d_quality;   //!
   TBranch        *b_cl3d_ipt;   //!
   TBranch        *b_cl3d_ienergy;   //!
   TBranch        *b_tower_n;   //!
   TBranch        *b_tower_pt;   //!
   TBranch        *b_tower_energy;   //!
   TBranch        *b_tower_eta;   //!
   TBranch        *b_tower_phi;   //!
   TBranch        *b_tower_etEm;   //!
   TBranch        *b_tower_etHad;   //!
   TBranch        *b_tower_iEta;   //!
   TBranch        *b_tower_iPhi;   //!
   TBranch        *b_cl3dfulltruth_n;   //!
   TBranch        *b_cl3dfulltruth_id;   //!
   TBranch        *b_cl3dfulltruth_pt;   //!
   TBranch        *b_cl3dfulltruth_energy;   //!
   TBranch        *b_cl3dfulltruth_eta;   //!
   TBranch        *b_cl3dfulltruth_phi;   //!
   TBranch        *b_cl3dfulltruth_clusters_n;   //!
   TBranch        *b_cl3dfulltruth_clusters_id;   //!
   TBranch        *b_cl3dfulltruth_showerlength;   //!
   TBranch        *b_cl3dfulltruth_coreshowerlength;   //!
   TBranch        *b_cl3dfulltruth_firstlayer;   //!
   TBranch        *b_cl3dfulltruth_maxlayer;   //!
   TBranch        *b_cl3dfulltruth_seetot;   //!
   TBranch        *b_cl3dfulltruth_seemax;   //!
   TBranch        *b_cl3dfulltruth_spptot;   //!
   TBranch        *b_cl3dfulltruth_sppmax;   //!
   TBranch        *b_cl3dfulltruth_szz;   //!
   TBranch        *b_cl3dfulltruth_srrtot;   //!
   TBranch        *b_cl3dfulltruth_srrmax;   //!
   TBranch        *b_cl3dfulltruth_srrmean;   //!
   TBranch        *b_cl3dfulltruth_emaxe;   //!
   TBranch        *b_cl3dfulltruth_hoe;   //!
   TBranch        *b_cl3dfulltruth_meanz;   //!
   TBranch        *b_cl3dfulltruth_layer10;   //!
   TBranch        *b_cl3dfulltruth_layer50;   //!
   TBranch        *b_cl3dfulltruth_layer90;   //!
   TBranch        *b_cl3dfulltruth_ntc67;   //!
   TBranch        *b_cl3dfulltruth_ntc90;   //!
   TBranch        *b_cl3dfulltruth_bdteg;   //!
   TBranch        *b_cl3dfulltruth_quality;   //!
   TBranch        *b_cl3dfulltruth_ipt;   //!
   TBranch        *b_cl3dfulltruth_ienergy;   //!
   TBranch        *b_tctruth_n;   //!
   TBranch        *b_tctruth_id;   //!
   TBranch        *b_tctruth_subdet;   //!
   TBranch        *b_tctruth_zside;   //!
   TBranch        *b_tctruth_layer;   //!
   TBranch        *b_tctruth_wafer;   //!
   TBranch        *b_tctruth_waferu;   //!
   TBranch        *b_tctruth_waferv;   //!
   TBranch        *b_tctruth_wafertype;   //!
   TBranch        *b_tctruth_panel_number;   //!
   TBranch        *b_tctruth_panel_sector;   //!
   TBranch        *b_tctruth_cell;   //!
   TBranch        *b_tctruth_cellu;   //!
   TBranch        *b_tctruth_cellv;   //!
   TBranch        *b_tctruth_data;   //!
   TBranch        *b_tctruth_uncompressedCharge;   //!
   TBranch        *b_tctruth_compressedCharge;   //!
   TBranch        *b_tctruth_pt;   //!
   TBranch        *b_tctruth_mipPt;   //!
   TBranch        *b_tctruth_energy;   //!
   TBranch        *b_tctruth_eta;   //!
   TBranch        *b_tctruth_phi;   //!
   TBranch        *b_tctruth_x;   //!
   TBranch        *b_tctruth_y;   //!
   TBranch        *b_tctruth_z;   //!
   TBranch        *b_tctruth_cluster_id;   //!
   TBranch        *b_tctruth_multicluster_id;   //!
   TBranch        *b_tctruth_multicluster_pt;   //!
   TBranch        *b_cl3dtruth_n;   //!
   TBranch        *b_cl3dtruth_id;   //!
   TBranch        *b_cl3dtruth_pt;   //!
   TBranch        *b_cl3dtruth_energy;   //!
   TBranch        *b_cl3dtruth_eta;   //!
   TBranch        *b_cl3dtruth_phi;   //!
   TBranch        *b_cl3dtruth_clusters_n;   //!
   TBranch        *b_cl3dtruth_clusters_id;   //!
   TBranch        *b_cl3dtruth_showerlength;   //!
   TBranch        *b_cl3dtruth_coreshowerlength;   //!
   TBranch        *b_cl3dtruth_firstlayer;   //!
   TBranch        *b_cl3dtruth_maxlayer;   //!
   TBranch        *b_cl3dtruth_seetot;   //!
   TBranch        *b_cl3dtruth_seemax;   //!
   TBranch        *b_cl3dtruth_spptot;   //!
   TBranch        *b_cl3dtruth_sppmax;   //!
   TBranch        *b_cl3dtruth_szz;   //!
   TBranch        *b_cl3dtruth_srrtot;   //!
   TBranch        *b_cl3dtruth_srrmax;   //!
   TBranch        *b_cl3dtruth_srrmean;   //!
   TBranch        *b_cl3dtruth_emaxe;   //!
   TBranch        *b_cl3dtruth_hoe;   //!
   TBranch        *b_cl3dtruth_meanz;   //!
   TBranch        *b_cl3dtruth_layer10;   //!
   TBranch        *b_cl3dtruth_layer50;   //!
   TBranch        *b_cl3dtruth_layer90;   //!
   TBranch        *b_cl3dtruth_ntc67;   //!
   TBranch        *b_cl3dtruth_ntc90;   //!
   TBranch        *b_cl3dtruth_bdteg;   //!
   TBranch        *b_cl3dtruth_quality;   //!
   TBranch        *b_cl3dtruth_ipt;   //!
   TBranch        *b_cl3dtruth_ienergy;   //!
   TBranch        *b_towertruth_n;   //!
   TBranch        *b_towertruth_pt;   //!
   TBranch        *b_towertruth_energy;   //!
   TBranch        *b_towertruth_eta;   //!
   TBranch        *b_towertruth_phi;   //!
   TBranch        *b_towertruth_etEm;   //!
   TBranch        *b_towertruth_etHad;   //!
   TBranch        *b_towertruth_iEta;   //!
   TBranch        *b_towertruth_iPhi;   //!

   bigTree (TChain * inputChain) : fChain(inputChain) { Init(fChain); }
   virtual ~bigTree() {}
   virtual Int_t GetEntry(Long64_t entry) { return fChain->GetEntry(entry); }

   void Init(TChain *inputChain)
   {
      // The Init() function is called when the selector needs to initialize
      // a new tree or chain. Typically here the branch addresses and branch
      // pointers of the tree will be set.
      // It is normally not necessary to make changes to the generated
      // code, but the routine can be extended by the user if needed.
      // Init() will be called many times when running on PROOF
      // (once per file to be processed).

      // Set object pointer
      gen_eta = 0;
      gen_phi = 0;
      gen_pt = 0;
      gen_energy = 0;
      gen_charge = 0;
      gen_pdgid = 0;
      gen_status = 0;
      gen_daughters = 0;
      genpart_eta = 0;
      genpart_phi = 0;
      genpart_pt = 0;
      genpart_energy = 0;
      genpart_dvx = 0;
      genpart_dvy = 0;
      genpart_dvz = 0;
      genpart_ovx = 0;
      genpart_ovy = 0;
      genpart_ovz = 0;
      genpart_mother = 0;
      genpart_exphi = 0;
      genpart_exeta = 0;
      genpart_exx = 0;
      genpart_exy = 0;
      genpart_fbrem = 0;
      genpart_pid = 0;
      genpart_gen = 0;
      genpart_reachedEE = 0;
      genpart_fromBeamPipe = 0;
      genpart_posx = 0;
      genpart_posy = 0;
      genpart_posz = 0;
      genjet_energy = 0;
      genjet_pt = 0;
      genjet_eta = 0;
      genjet_phi = 0;
      gentau_pt = 0;
      gentau_eta = 0;
      gentau_phi = 0;
      gentau_energy = 0;
      gentau_mass = 0;
      gentau_vis_pt = 0;
      gentau_vis_eta = 0;
      gentau_vis_phi = 0;
      gentau_vis_energy = 0;
      gentau_vis_mass = 0;
      gentau_products_pt = 0;
      gentau_products_eta = 0;
      gentau_products_phi = 0;
      gentau_products_energy = 0;
      gentau_products_mass = 0;
      gentau_products_id = 0;
      gentau_decayMode = 0;
      gentau_totNproducts = 0;
      gentau_totNgamma = 0;
      gentau_totNpiZero = 0;
      gentau_totNcharged = 0;
      hgcdigi_id = 0;
      hgcdigi_subdet = 0;
      hgcdigi_zside = 0;
      hgcdigi_layer = 0;
      hgcdigi_wafertype = 0;
      hgcdigi_eta = 0;
      hgcdigi_phi = 0;
      hgcdigi_z = 0;
      hgcdigi_data_BX2 = 0;
      hgcdigi_isadc_BX2 = 0;
      hgcdigi_waferu = 0;
      hgcdigi_waferv = 0;
      hgcdigi_cellu = 0;
      hgcdigi_cellv = 0;
      hgcdigi_wafer = 0;
      hgcdigi_cell = 0;
      bhdigi_id = 0;
      bhdigi_subdet = 0;
      bhdigi_zside = 0;
      bhdigi_layer = 0;
      bhdigi_ieta = 0;
      bhdigi_iphi = 0;
      bhdigi_eta = 0;
      bhdigi_phi = 0;
      bhdigi_z = 0;
      bhdigi_data_BX2 = 0;
      bhdigi_isadc_BX2 = 0;
      tc_id = 0;
      tc_subdet = 0;
      tc_zside = 0;
      tc_layer = 0;
      tc_wafer = 0;
      tc_waferu = 0;
      tc_waferv = 0;
      tc_wafertype = 0;
      tc_panel_number = 0;
      tc_panel_sector = 0;
      tc_cell = 0;
      tc_cellu = 0;
      tc_cellv = 0;
      tc_data = 0;
      tc_uncompressedCharge = 0;
      tc_compressedCharge = 0;
      tc_pt = 0;
      tc_mipPt = 0;
      tc_energy = 0;
      tc_eta = 0;
      tc_phi = 0;
      tc_x = 0;
      tc_y = 0;
      tc_z = 0;
      tc_cluster_id = 0;
      tc_multicluster_id = 0;
      tc_multicluster_pt = 0;
      tc_genparticle_index = 0;
      ts_id = 0;
      ts_subdet = 0;
      ts_zside = 0;
      ts_layer = 0;
      ts_wafer = 0;
      ts_wafertype = 0;
      ts_panel_number = 0;
      ts_panel_sector = 0;
      ts_data = 0;
      ts_pt = 0;
      ts_mipPt = 0;
      ts_energy = 0;
      ts_eta = 0;
      ts_phi = 0;
      ts_x = 0;
      ts_y = 0;
      ts_z = 0;
      cl3d_id = 0;
      cl3d_pt = 0;
      cl3d_energy = 0;
      cl3d_eta = 0;
      cl3d_phi = 0;
      cl3d_clusters_n = 0;
      cl3d_clusters_id = 0;
      cl3d_showerlength = 0;
      cl3d_coreshowerlength = 0;
      cl3d_firstlayer = 0;
      cl3d_maxlayer = 0;
      cl3d_seetot = 0;
      cl3d_seemax = 0;
      cl3d_spptot = 0;
      cl3d_sppmax = 0;
      cl3d_szz = 0;
      cl3d_srrtot = 0;
      cl3d_srrmax = 0;
      cl3d_srrmean = 0;
      cl3d_emaxe = 0;
      cl3d_hoe = 0;
      cl3d_meanz = 0;
      cl3d_layer10 = 0;
      cl3d_layer50 = 0;
      cl3d_layer90 = 0;
      cl3d_ntc67 = 0;
      cl3d_ntc90 = 0;
      cl3d_bdteg = 0;
      cl3d_quality = 0;
      cl3d_ipt = 0;
      cl3d_ienergy = 0;
      tower_pt = 0;
      tower_energy = 0;
      tower_eta = 0;
      tower_phi = 0;
      tower_etEm = 0;
      tower_etHad = 0;
      tower_iEta = 0;
      tower_iPhi = 0;
      cl3dfulltruth_id = 0;
      cl3dfulltruth_pt = 0;
      cl3dfulltruth_energy = 0;
      cl3dfulltruth_eta = 0;
      cl3dfulltruth_phi = 0;
      cl3dfulltruth_clusters_n = 0;
      cl3dfulltruth_clusters_id = 0;
      cl3dfulltruth_showerlength = 0;
      cl3dfulltruth_coreshowerlength = 0;
      cl3dfulltruth_firstlayer = 0;
      cl3dfulltruth_maxlayer = 0;
      cl3dfulltruth_seetot = 0;
      cl3dfulltruth_seemax = 0;
      cl3dfulltruth_spptot = 0;
      cl3dfulltruth_sppmax = 0;
      cl3dfulltruth_szz = 0;
      cl3dfulltruth_srrtot = 0;
      cl3dfulltruth_srrmax = 0;
      cl3dfulltruth_srrmean = 0;
      cl3dfulltruth_emaxe = 0;
      cl3dfulltruth_hoe = 0;
      cl3dfulltruth_meanz = 0;
      cl3dfulltruth_layer10 = 0;
      cl3dfulltruth_layer50 = 0;
      cl3dfulltruth_layer90 = 0;
      cl3dfulltruth_ntc67 = 0;
      cl3dfulltruth_ntc90 = 0;
      cl3dfulltruth_bdteg = 0;
      cl3dfulltruth_quality = 0;
      cl3dfulltruth_ipt = 0;
      cl3dfulltruth_ienergy = 0;
      tctruth_id = 0;
      tctruth_subdet = 0;
      tctruth_zside = 0;
      tctruth_layer = 0;
      tctruth_wafer = 0;
      tctruth_waferu = 0;
      tctruth_waferv = 0;
      tctruth_wafertype = 0;
      tctruth_panel_number = 0;
      tctruth_panel_sector = 0;
      tctruth_cell = 0;
      tctruth_cellu = 0;
      tctruth_cellv = 0;
      tctruth_data = 0;
      tctruth_uncompressedCharge = 0;
      tctruth_compressedCharge = 0;
      tctruth_pt = 0;
      tctruth_mipPt = 0;
      tctruth_energy = 0;
      tctruth_eta = 0;
      tctruth_phi = 0;
      tctruth_x = 0;
      tctruth_y = 0;
      tctruth_z = 0;
      tctruth_cluster_id = 0;
      tctruth_multicluster_id = 0;
      tctruth_multicluster_pt = 0;
      cl3dtruth_id = 0;
      cl3dtruth_pt = 0;
      cl3dtruth_energy = 0;
      cl3dtruth_eta = 0;
      cl3dtruth_phi = 0;
      cl3dtruth_clusters_n = 0;
      cl3dtruth_clusters_id = 0;
      cl3dtruth_showerlength = 0;
      cl3dtruth_coreshowerlength = 0;
      cl3dtruth_firstlayer = 0;
      cl3dtruth_maxlayer = 0;
      cl3dtruth_seetot = 0;
      cl3dtruth_seemax = 0;
      cl3dtruth_spptot = 0;
      cl3dtruth_sppmax = 0;
      cl3dtruth_szz = 0;
      cl3dtruth_srrtot = 0;
      cl3dtruth_srrmax = 0;
      cl3dtruth_srrmean = 0;
      cl3dtruth_emaxe = 0;
      cl3dtruth_hoe = 0;
      cl3dtruth_meanz = 0;
      cl3dtruth_layer10 = 0;
      cl3dtruth_layer50 = 0;
      cl3dtruth_layer90 = 0;
      cl3dtruth_ntc67 = 0;
      cl3dtruth_ntc90 = 0;
      cl3dtruth_bdteg = 0;
      cl3dtruth_quality = 0;
      cl3dtruth_ipt = 0;
      cl3dtruth_ienergy = 0;
      towertruth_pt = 0;
      towertruth_energy = 0;
      towertruth_eta = 0;
      towertruth_phi = 0;
      towertruth_etEm = 0;
      towertruth_etHad = 0;
      towertruth_iEta = 0;
      towertruth_iPhi = 0;
      
      fChain->SetMakeClass(1);

      fChain->SetBranchAddress("run", &run, &b_run);
      fChain->SetBranchAddress("event", &event, &b_event);
      fChain->SetBranchAddress("lumi", &lumi, &b_lumi);
      fChain->SetBranchAddress("gen_n", &gen_n, &b_gen_n);
      fChain->SetBranchAddress("gen_PUNumInt", &gen_PUNumInt, &b_gen_PUNumInt);
      fChain->SetBranchAddress("gen_TrueNumInt", &gen_TrueNumInt, &b_gen_TrueNumInt);
      fChain->SetBranchAddress("vtx_x", &vtx_x, &b_vtx_x);
      fChain->SetBranchAddress("vtx_y", &vtx_y, &b_vtx_y);
      fChain->SetBranchAddress("vtx_z", &vtx_z, &b_vtx_z);
      fChain->SetBranchAddress("gen_eta", &gen_eta, &b_gen_eta);
      fChain->SetBranchAddress("gen_phi", &gen_phi, &b_gen_phi);
      fChain->SetBranchAddress("gen_pt", &gen_pt, &b_gen_pt);
      fChain->SetBranchAddress("gen_energy", &gen_energy, &b_gen_energy);
      fChain->SetBranchAddress("gen_charge", &gen_charge, &b_gen_charge);
      fChain->SetBranchAddress("gen_pdgid", &gen_pdgid, &b_gen_pdgid);
      fChain->SetBranchAddress("gen_status", &gen_status, &b_gen_status);
      fChain->SetBranchAddress("gen_daughters", &gen_daughters, &b_gen_daughters);
      fChain->SetBranchAddress("genpart_eta", &genpart_eta, &b_genpart_eta);
      fChain->SetBranchAddress("genpart_phi", &genpart_phi, &b_genpart_phi);
      fChain->SetBranchAddress("genpart_pt", &genpart_pt, &b_genpart_pt);
      fChain->SetBranchAddress("genpart_energy", &genpart_energy, &b_genpart_energy);
      fChain->SetBranchAddress("genpart_dvx", &genpart_dvx, &b_genpart_dvx);
      fChain->SetBranchAddress("genpart_dvy", &genpart_dvy, &b_genpart_dvy);
      fChain->SetBranchAddress("genpart_dvz", &genpart_dvz, &b_genpart_dvz);
      fChain->SetBranchAddress("genpart_ovx", &genpart_ovx, &b_genpart_ovx);
      fChain->SetBranchAddress("genpart_ovy", &genpart_ovy, &b_genpart_ovy);
      fChain->SetBranchAddress("genpart_ovz", &genpart_ovz, &b_genpart_ovz);
      fChain->SetBranchAddress("genpart_mother", &genpart_mother, &b_genpart_mother);
      fChain->SetBranchAddress("genpart_exphi", &genpart_exphi, &b_genpart_exphi);
      fChain->SetBranchAddress("genpart_exeta", &genpart_exeta, &b_genpart_exeta);
      fChain->SetBranchAddress("genpart_exx", &genpart_exx, &b_genpart_exx);
      fChain->SetBranchAddress("genpart_exy", &genpart_exy, &b_genpart_exy);
      fChain->SetBranchAddress("genpart_fbrem", &genpart_fbrem, &b_genpart_fbrem);
      fChain->SetBranchAddress("genpart_pid", &genpart_pid, &b_genpart_pid);
      fChain->SetBranchAddress("genpart_gen", &genpart_gen, &b_genpart_gen);
      fChain->SetBranchAddress("genpart_reachedEE", &genpart_reachedEE, &b_genpart_reachedEE);
      fChain->SetBranchAddress("genpart_fromBeamPipe", &genpart_fromBeamPipe, &b_genpart_fromBeamPipe);
      fChain->SetBranchAddress("genpart_posx", &genpart_posx, &b_genpart_posx);
      fChain->SetBranchAddress("genpart_posy", &genpart_posy, &b_genpart_posy);
      fChain->SetBranchAddress("genpart_posz", &genpart_posz, &b_genpart_posz);
      fChain->SetBranchAddress("genjet_n", &genjet_n, &b_genjet_n);
      fChain->SetBranchAddress("genjet_energy", &genjet_energy, &b_genjet_energy);
      fChain->SetBranchAddress("genjet_pt", &genjet_pt, &b_genjet_pt);
      fChain->SetBranchAddress("genjet_eta", &genjet_eta, &b_genjet_eta);
      fChain->SetBranchAddress("genjet_phi", &genjet_phi, &b_genjet_phi);
      fChain->SetBranchAddress("gentau_pt", &gentau_pt, &b_gentau_pt);
      fChain->SetBranchAddress("gentau_eta", &gentau_eta, &b_gentau_eta);
      fChain->SetBranchAddress("gentau_phi", &gentau_phi, &b_gentau_phi);
      fChain->SetBranchAddress("gentau_energy", &gentau_energy, &b_gentau_energy);
      fChain->SetBranchAddress("gentau_mass", &gentau_mass, &b_gentau_mass);
      fChain->SetBranchAddress("gentau_vis_pt", &gentau_vis_pt, &b_gentau_vis_pt);
      fChain->SetBranchAddress("gentau_vis_eta", &gentau_vis_eta, &b_gentau_vis_eta);
      fChain->SetBranchAddress("gentau_vis_phi", &gentau_vis_phi, &b_gentau_vis_phi);
      fChain->SetBranchAddress("gentau_vis_energy", &gentau_vis_energy, &b_gentau_vis_energy);
      fChain->SetBranchAddress("gentau_vis_mass", &gentau_vis_mass, &b_gentau_vis_mass);
      fChain->SetBranchAddress("gentau_products_pt", &gentau_products_pt, &b_gentau_products_pt);
      fChain->SetBranchAddress("gentau_products_eta", &gentau_products_eta, &b_gentau_products_eta);
      fChain->SetBranchAddress("gentau_products_phi", &gentau_products_phi, &b_gentau_products_phi);
      fChain->SetBranchAddress("gentau_products_energy", &gentau_products_energy, &b_gentau_products_energy);
      fChain->SetBranchAddress("gentau_products_mass", &gentau_products_mass, &b_gentau_products_mass);
      fChain->SetBranchAddress("gentau_products_id", &gentau_products_id, &b_gentau_products_id);
      fChain->SetBranchAddress("gentau_decayMode", &gentau_decayMode, &b_gentau_decayMode);
      fChain->SetBranchAddress("gentau_totNproducts", &gentau_totNproducts, &b_gentau_totNproducts);
      fChain->SetBranchAddress("gentau_totNgamma", &gentau_totNgamma, &b_gentau_totNgamma);
      fChain->SetBranchAddress("gentau_totNpiZero", &gentau_totNpiZero, &b_gentau_totNpiZero);
      fChain->SetBranchAddress("gentau_totNcharged", &gentau_totNcharged, &b_gentau_totNcharged);
      fChain->SetBranchAddress("hgcdigi_n", &hgcdigi_n, &b_hgcdigi_n);
      fChain->SetBranchAddress("hgcdigi_id", &hgcdigi_id, &b_hgcdigi_id);
      fChain->SetBranchAddress("hgcdigi_subdet", &hgcdigi_subdet, &b_hgcdigi_subdet);
      fChain->SetBranchAddress("hgcdigi_zside", &hgcdigi_zside, &b_hgcdigi_zside);
      fChain->SetBranchAddress("hgcdigi_layer", &hgcdigi_layer, &b_hgcdigi_layer);
      fChain->SetBranchAddress("hgcdigi_wafertype", &hgcdigi_wafertype, &b_hgcdigi_wafertype);
      fChain->SetBranchAddress("hgcdigi_eta", &hgcdigi_eta, &b_hgcdigi_eta);
      fChain->SetBranchAddress("hgcdigi_phi", &hgcdigi_phi, &b_hgcdigi_phi);
      fChain->SetBranchAddress("hgcdigi_z", &hgcdigi_z, &b_hgcdigi_z);
      fChain->SetBranchAddress("hgcdigi_data_BX2", &hgcdigi_data_BX2, &b_hgcdigi_data_BX2);
      fChain->SetBranchAddress("hgcdigi_isadc_BX2", &hgcdigi_isadc_BX2, &b_hgcdigi_isadc_BX2);
      fChain->SetBranchAddress("hgcdigi_waferu", &hgcdigi_waferu, &b_hgcdigi_waferu);
      fChain->SetBranchAddress("hgcdigi_waferv", &hgcdigi_waferv, &b_hgcdigi_waferv);
      fChain->SetBranchAddress("hgcdigi_cellu", &hgcdigi_cellu, &b_hgcdigi_cellu);
      fChain->SetBranchAddress("hgcdigi_cellv", &hgcdigi_cellv, &b_hgcdigi_cellv);
      fChain->SetBranchAddress("hgcdigi_wafer", &hgcdigi_wafer, &b_hgcdigi_wafer);
      fChain->SetBranchAddress("hgcdigi_cell", &hgcdigi_cell, &b_hgcdigi_cell);
      fChain->SetBranchAddress("bhdigi_n", &bhdigi_n, &b_bhdigi_n);
      fChain->SetBranchAddress("bhdigi_id", &bhdigi_id, &b_bhdigi_id);
      fChain->SetBranchAddress("bhdigi_subdet", &bhdigi_subdet, &b_bhdigi_subdet);
      fChain->SetBranchAddress("bhdigi_zside", &bhdigi_zside, &b_bhdigi_zside);
      fChain->SetBranchAddress("bhdigi_layer", &bhdigi_layer, &b_bhdigi_layer);
      fChain->SetBranchAddress("bhdigi_ieta", &bhdigi_ieta, &b_bhdigi_ieta);
      fChain->SetBranchAddress("bhdigi_iphi", &bhdigi_iphi, &b_bhdigi_iphi);
      fChain->SetBranchAddress("bhdigi_eta", &bhdigi_eta, &b_bhdigi_eta);
      fChain->SetBranchAddress("bhdigi_phi", &bhdigi_phi, &b_bhdigi_phi);
      fChain->SetBranchAddress("bhdigi_z", &bhdigi_z, &b_bhdigi_z);
      fChain->SetBranchAddress("bhdigi_data_BX2", &bhdigi_data_BX2, &b_bhdigi_data_BX2);
      fChain->SetBranchAddress("bhdigi_isadc_BX2", &bhdigi_isadc_BX2, &b_bhdigi_isadc_BX2);
      fChain->SetBranchAddress("tc_n", &tc_n, &b_tc_n);
      fChain->SetBranchAddress("tc_id", &tc_id, &b_tc_id);
      fChain->SetBranchAddress("tc_subdet", &tc_subdet, &b_tc_subdet);
      fChain->SetBranchAddress("tc_zside", &tc_zside, &b_tc_zside);
      fChain->SetBranchAddress("tc_layer", &tc_layer, &b_tc_layer);
      fChain->SetBranchAddress("tc_wafer", &tc_wafer, &b_tc_wafer);
      fChain->SetBranchAddress("tc_waferu", &tc_waferu, &b_tc_waferu);
      fChain->SetBranchAddress("tc_waferv", &tc_waferv, &b_tc_waferv);
      fChain->SetBranchAddress("tc_wafertype", &tc_wafertype, &b_tc_wafertype);
      fChain->SetBranchAddress("tc_panel_number", &tc_panel_number, &b_tc_panel_number);
      fChain->SetBranchAddress("tc_panel_sector", &tc_panel_sector, &b_tc_panel_sector);
      fChain->SetBranchAddress("tc_cell", &tc_cell, &b_tc_cell);
      fChain->SetBranchAddress("tc_cellu", &tc_cellu, &b_tc_cellu);
      fChain->SetBranchAddress("tc_cellv", &tc_cellv, &b_tc_cellv);
      fChain->SetBranchAddress("tc_data", &tc_data, &b_tc_data);
      fChain->SetBranchAddress("tc_uncompressedCharge", &tc_uncompressedCharge, &b_tc_uncompressedCharge);
      fChain->SetBranchAddress("tc_compressedCharge", &tc_compressedCharge, &b_tc_compressedCharge);
      fChain->SetBranchAddress("tc_pt", &tc_pt, &b_tc_pt);
      fChain->SetBranchAddress("tc_mipPt", &tc_mipPt, &b_tc_mipPt);
      fChain->SetBranchAddress("tc_energy", &tc_energy, &b_tc_energy);
      fChain->SetBranchAddress("tc_eta", &tc_eta, &b_tc_eta);
      fChain->SetBranchAddress("tc_phi", &tc_phi, &b_tc_phi);
      fChain->SetBranchAddress("tc_x", &tc_x, &b_tc_x);
      fChain->SetBranchAddress("tc_y", &tc_y, &b_tc_y);
      fChain->SetBranchAddress("tc_z", &tc_z, &b_tc_z);
      fChain->SetBranchAddress("tc_cluster_id", &tc_cluster_id, &b_tc_cluster_id);
      fChain->SetBranchAddress("tc_multicluster_id", &tc_multicluster_id, &b_tc_multicluster_id);
      fChain->SetBranchAddress("tc_multicluster_pt", &tc_multicluster_pt, &b_tc_multicluster_pt);
      fChain->SetBranchAddress("tc_genparticle_index", &tc_genparticle_index, &b_tc_genparticle_index);
      fChain->SetBranchAddress("ts_n", &ts_n, &b_ts_n);
      fChain->SetBranchAddress("ts_id", &ts_id, &b_ts_id);
      fChain->SetBranchAddress("ts_subdet", &ts_subdet, &b_ts_subdet);
      fChain->SetBranchAddress("ts_zside", &ts_zside, &b_ts_zside);
      fChain->SetBranchAddress("ts_layer", &ts_layer, &b_ts_layer);
      fChain->SetBranchAddress("ts_wafer", &ts_wafer, &b_ts_wafer);
      fChain->SetBranchAddress("ts_wafertype", &ts_wafertype, &b_ts_wafertype);
      fChain->SetBranchAddress("ts_panel_number", &ts_panel_number, &b_ts_panel_number);
      fChain->SetBranchAddress("ts_panel_sector", &ts_panel_sector, &b_ts_panel_sector);
      fChain->SetBranchAddress("ts_data", &ts_data, &b_ts_data);
      fChain->SetBranchAddress("ts_pt", &ts_pt, &b_ts_pt);
      fChain->SetBranchAddress("ts_mipPt", &ts_mipPt, &b_ts_mipPt);
      fChain->SetBranchAddress("ts_energy", &ts_energy, &b_ts_energy);
      fChain->SetBranchAddress("ts_eta", &ts_eta, &b_ts_eta);
      fChain->SetBranchAddress("ts_phi", &ts_phi, &b_ts_phi);
      fChain->SetBranchAddress("ts_x", &ts_x, &b_ts_x);
      fChain->SetBranchAddress("ts_y", &ts_y, &b_ts_y);
      fChain->SetBranchAddress("ts_z", &ts_z, &b_ts_z);
      fChain->SetBranchAddress("cl3d_n", &cl3d_n, &b_cl3d_n);
      fChain->SetBranchAddress("cl3d_id", &cl3d_id, &b_cl3d_id);
      fChain->SetBranchAddress("cl3d_pt", &cl3d_pt, &b_cl3d_pt);
      fChain->SetBranchAddress("cl3d_energy", &cl3d_energy, &b_cl3d_energy);
      fChain->SetBranchAddress("cl3d_eta", &cl3d_eta, &b_cl3d_eta);
      fChain->SetBranchAddress("cl3d_phi", &cl3d_phi, &b_cl3d_phi);
      fChain->SetBranchAddress("cl3d_clusters_n", &cl3d_clusters_n, &b_cl3d_clusters_n);
      fChain->SetBranchAddress("cl3d_clusters_id", &cl3d_clusters_id, &b_cl3d_clusters_id);
      fChain->SetBranchAddress("cl3d_showerlength", &cl3d_showerlength, &b_cl3d_showerlength);
      fChain->SetBranchAddress("cl3d_coreshowerlength", &cl3d_coreshowerlength, &b_cl3d_coreshowerlength);
      fChain->SetBranchAddress("cl3d_firstlayer", &cl3d_firstlayer, &b_cl3d_firstlayer);
      fChain->SetBranchAddress("cl3d_maxlayer", &cl3d_maxlayer, &b_cl3d_maxlayer);
      fChain->SetBranchAddress("cl3d_seetot", &cl3d_seetot, &b_cl3d_seetot);
      fChain->SetBranchAddress("cl3d_seemax", &cl3d_seemax, &b_cl3d_seemax);
      fChain->SetBranchAddress("cl3d_spptot", &cl3d_spptot, &b_cl3d_spptot);
      fChain->SetBranchAddress("cl3d_sppmax", &cl3d_sppmax, &b_cl3d_sppmax);
      fChain->SetBranchAddress("cl3d_szz", &cl3d_szz, &b_cl3d_szz);
      fChain->SetBranchAddress("cl3d_srrtot", &cl3d_srrtot, &b_cl3d_srrtot);
      fChain->SetBranchAddress("cl3d_srrmax", &cl3d_srrmax, &b_cl3d_srrmax);
      fChain->SetBranchAddress("cl3d_srrmean", &cl3d_srrmean, &b_cl3d_srrmean);
      fChain->SetBranchAddress("cl3d_emaxe", &cl3d_emaxe, &b_cl3d_emaxe);
      fChain->SetBranchAddress("cl3d_hoe", &cl3d_hoe, &b_cl3d_hoe);
      fChain->SetBranchAddress("cl3d_meanz", &cl3d_meanz, &b_cl3d_meanz);
      fChain->SetBranchAddress("cl3d_layer10", &cl3d_layer10, &b_cl3d_layer10);
      fChain->SetBranchAddress("cl3d_layer50", &cl3d_layer50, &b_cl3d_layer50);
      fChain->SetBranchAddress("cl3d_layer90", &cl3d_layer90, &b_cl3d_layer90);
      fChain->SetBranchAddress("cl3d_ntc67", &cl3d_ntc67, &b_cl3d_ntc67);
      fChain->SetBranchAddress("cl3d_ntc90", &cl3d_ntc90, &b_cl3d_ntc90);
      fChain->SetBranchAddress("cl3d_bdteg", &cl3d_bdteg, &b_cl3d_bdteg);
      fChain->SetBranchAddress("cl3d_quality", &cl3d_quality, &b_cl3d_quality);
      fChain->SetBranchAddress("cl3d_ipt", &cl3d_ipt, &b_cl3d_ipt);
      fChain->SetBranchAddress("cl3d_ienergy", &cl3d_ienergy, &b_cl3d_ienergy);
      fChain->SetBranchAddress("tower_n", &tower_n, &b_tower_n);
      fChain->SetBranchAddress("tower_pt", &tower_pt, &b_tower_pt);
      fChain->SetBranchAddress("tower_energy", &tower_energy, &b_tower_energy);
      fChain->SetBranchAddress("tower_eta", &tower_eta, &b_tower_eta);
      fChain->SetBranchAddress("tower_phi", &tower_phi, &b_tower_phi);
      fChain->SetBranchAddress("tower_etEm", &tower_etEm, &b_tower_etEm);
      fChain->SetBranchAddress("tower_etHad", &tower_etHad, &b_tower_etHad);
      fChain->SetBranchAddress("tower_iEta", &tower_iEta, &b_tower_iEta);
      fChain->SetBranchAddress("tower_iPhi", &tower_iPhi, &b_tower_iPhi);
      fChain->SetBranchAddress("cl3dfulltruth_n", &cl3dfulltruth_n, &b_cl3dfulltruth_n);
      fChain->SetBranchAddress("cl3dfulltruth_id", &cl3dfulltruth_id, &b_cl3dfulltruth_id);
      fChain->SetBranchAddress("cl3dfulltruth_pt", &cl3dfulltruth_pt, &b_cl3dfulltruth_pt);
      fChain->SetBranchAddress("cl3dfulltruth_energy", &cl3dfulltruth_energy, &b_cl3dfulltruth_energy);
      fChain->SetBranchAddress("cl3dfulltruth_eta", &cl3dfulltruth_eta, &b_cl3dfulltruth_eta);
      fChain->SetBranchAddress("cl3dfulltruth_phi", &cl3dfulltruth_phi, &b_cl3dfulltruth_phi);
      fChain->SetBranchAddress("cl3dfulltruth_clusters_n", &cl3dfulltruth_clusters_n, &b_cl3dfulltruth_clusters_n);
      fChain->SetBranchAddress("cl3dfulltruth_clusters_id", &cl3dfulltruth_clusters_id, &b_cl3dfulltruth_clusters_id);
      fChain->SetBranchAddress("cl3dfulltruth_showerlength", &cl3dfulltruth_showerlength, &b_cl3dfulltruth_showerlength);
      fChain->SetBranchAddress("cl3dfulltruth_coreshowerlength", &cl3dfulltruth_coreshowerlength, &b_cl3dfulltruth_coreshowerlength);
      fChain->SetBranchAddress("cl3dfulltruth_firstlayer", &cl3dfulltruth_firstlayer, &b_cl3dfulltruth_firstlayer);
      fChain->SetBranchAddress("cl3dfulltruth_maxlayer", &cl3dfulltruth_maxlayer, &b_cl3dfulltruth_maxlayer);
      fChain->SetBranchAddress("cl3dfulltruth_seetot", &cl3dfulltruth_seetot, &b_cl3dfulltruth_seetot);
      fChain->SetBranchAddress("cl3dfulltruth_seemax", &cl3dfulltruth_seemax, &b_cl3dfulltruth_seemax);
      fChain->SetBranchAddress("cl3dfulltruth_spptot", &cl3dfulltruth_spptot, &b_cl3dfulltruth_spptot);
      fChain->SetBranchAddress("cl3dfulltruth_sppmax", &cl3dfulltruth_sppmax, &b_cl3dfulltruth_sppmax);
      fChain->SetBranchAddress("cl3dfulltruth_szz", &cl3dfulltruth_szz, &b_cl3dfulltruth_szz);
      fChain->SetBranchAddress("cl3dfulltruth_srrtot", &cl3dfulltruth_srrtot, &b_cl3dfulltruth_srrtot);
      fChain->SetBranchAddress("cl3dfulltruth_srrmax", &cl3dfulltruth_srrmax, &b_cl3dfulltruth_srrmax);
      fChain->SetBranchAddress("cl3dfulltruth_srrmean", &cl3dfulltruth_srrmean, &b_cl3dfulltruth_srrmean);
      fChain->SetBranchAddress("cl3dfulltruth_emaxe", &cl3dfulltruth_emaxe, &b_cl3dfulltruth_emaxe);
      fChain->SetBranchAddress("cl3dfulltruth_hoe", &cl3dfulltruth_hoe, &b_cl3dfulltruth_hoe);
      fChain->SetBranchAddress("cl3dfulltruth_meanz", &cl3dfulltruth_meanz, &b_cl3dfulltruth_meanz);
      fChain->SetBranchAddress("cl3dfulltruth_layer10", &cl3dfulltruth_layer10, &b_cl3dfulltruth_layer10);
      fChain->SetBranchAddress("cl3dfulltruth_layer50", &cl3dfulltruth_layer50, &b_cl3dfulltruth_layer50);
      fChain->SetBranchAddress("cl3dfulltruth_layer90", &cl3dfulltruth_layer90, &b_cl3dfulltruth_layer90);
      fChain->SetBranchAddress("cl3dfulltruth_ntc67", &cl3dfulltruth_ntc67, &b_cl3dfulltruth_ntc67);
      fChain->SetBranchAddress("cl3dfulltruth_ntc90", &cl3dfulltruth_ntc90, &b_cl3dfulltruth_ntc90);
      fChain->SetBranchAddress("cl3dfulltruth_bdteg", &cl3dfulltruth_bdteg, &b_cl3dfulltruth_bdteg);
      fChain->SetBranchAddress("cl3dfulltruth_quality", &cl3dfulltruth_quality, &b_cl3dfulltruth_quality);
      fChain->SetBranchAddress("cl3dfulltruth_ipt", &cl3dfulltruth_ipt, &b_cl3dfulltruth_ipt);
      fChain->SetBranchAddress("cl3dfulltruth_ienergy", &cl3dfulltruth_ienergy, &b_cl3dfulltruth_ienergy);
      fChain->SetBranchAddress("tctruth_n", &tctruth_n, &b_tctruth_n);
      fChain->SetBranchAddress("tctruth_id", &tctruth_id, &b_tctruth_id);
      fChain->SetBranchAddress("tctruth_subdet", &tctruth_subdet, &b_tctruth_subdet);
      fChain->SetBranchAddress("tctruth_zside", &tctruth_zside, &b_tctruth_zside);
      fChain->SetBranchAddress("tctruth_layer", &tctruth_layer, &b_tctruth_layer);
      fChain->SetBranchAddress("tctruth_wafer", &tctruth_wafer, &b_tctruth_wafer);
      fChain->SetBranchAddress("tctruth_waferu", &tctruth_waferu, &b_tctruth_waferu);
      fChain->SetBranchAddress("tctruth_waferv", &tctruth_waferv, &b_tctruth_waferv);
      fChain->SetBranchAddress("tctruth_wafertype", &tctruth_wafertype, &b_tctruth_wafertype);
      fChain->SetBranchAddress("tctruth_panel_number", &tctruth_panel_number, &b_tctruth_panel_number);
      fChain->SetBranchAddress("tctruth_panel_sector", &tctruth_panel_sector, &b_tctruth_panel_sector);
      fChain->SetBranchAddress("tctruth_cell", &tctruth_cell, &b_tctruth_cell);
      fChain->SetBranchAddress("tctruth_cellu", &tctruth_cellu, &b_tctruth_cellu);
      fChain->SetBranchAddress("tctruth_cellv", &tctruth_cellv, &b_tctruth_cellv);
      fChain->SetBranchAddress("tctruth_data", &tctruth_data, &b_tctruth_data);
      fChain->SetBranchAddress("tctruth_uncompressedCharge", &tctruth_uncompressedCharge, &b_tctruth_uncompressedCharge);
      fChain->SetBranchAddress("tctruth_compressedCharge", &tctruth_compressedCharge, &b_tctruth_compressedCharge);
      fChain->SetBranchAddress("tctruth_pt", &tctruth_pt, &b_tctruth_pt);
      fChain->SetBranchAddress("tctruth_mipPt", &tctruth_mipPt, &b_tctruth_mipPt);
      fChain->SetBranchAddress("tctruth_energy", &tctruth_energy, &b_tctruth_energy);
      fChain->SetBranchAddress("tctruth_eta", &tctruth_eta, &b_tctruth_eta);
      fChain->SetBranchAddress("tctruth_phi", &tctruth_phi, &b_tctruth_phi);
      fChain->SetBranchAddress("tctruth_x", &tctruth_x, &b_tctruth_x);
      fChain->SetBranchAddress("tctruth_y", &tctruth_y, &b_tctruth_y);
      fChain->SetBranchAddress("tctruth_z", &tctruth_z, &b_tctruth_z);
      fChain->SetBranchAddress("tctruth_cluster_id", &tctruth_cluster_id, &b_tctruth_cluster_id);
      fChain->SetBranchAddress("tctruth_multicluster_id", &tctruth_multicluster_id, &b_tctruth_multicluster_id);
      fChain->SetBranchAddress("tctruth_multicluster_pt", &tctruth_multicluster_pt, &b_tctruth_multicluster_pt);
      fChain->SetBranchAddress("cl3dtruth_n", &cl3dtruth_n, &b_cl3dtruth_n);
      fChain->SetBranchAddress("cl3dtruth_id", &cl3dtruth_id, &b_cl3dtruth_id);
      fChain->SetBranchAddress("cl3dtruth_pt", &cl3dtruth_pt, &b_cl3dtruth_pt);
      fChain->SetBranchAddress("cl3dtruth_energy", &cl3dtruth_energy, &b_cl3dtruth_energy);
      fChain->SetBranchAddress("cl3dtruth_eta", &cl3dtruth_eta, &b_cl3dtruth_eta);
      fChain->SetBranchAddress("cl3dtruth_phi", &cl3dtruth_phi, &b_cl3dtruth_phi);
      fChain->SetBranchAddress("cl3dtruth_clusters_n", &cl3dtruth_clusters_n, &b_cl3dtruth_clusters_n);
      fChain->SetBranchAddress("cl3dtruth_clusters_id", &cl3dtruth_clusters_id, &b_cl3dtruth_clusters_id);
      fChain->SetBranchAddress("cl3dtruth_showerlength", &cl3dtruth_showerlength, &b_cl3dtruth_showerlength);
      fChain->SetBranchAddress("cl3dtruth_coreshowerlength", &cl3dtruth_coreshowerlength, &b_cl3dtruth_coreshowerlength);
      fChain->SetBranchAddress("cl3dtruth_firstlayer", &cl3dtruth_firstlayer, &b_cl3dtruth_firstlayer);
      fChain->SetBranchAddress("cl3dtruth_maxlayer", &cl3dtruth_maxlayer, &b_cl3dtruth_maxlayer);
      fChain->SetBranchAddress("cl3dtruth_seetot", &cl3dtruth_seetot, &b_cl3dtruth_seetot);
      fChain->SetBranchAddress("cl3dtruth_seemax", &cl3dtruth_seemax, &b_cl3dtruth_seemax);
      fChain->SetBranchAddress("cl3dtruth_spptot", &cl3dtruth_spptot, &b_cl3dtruth_spptot);
      fChain->SetBranchAddress("cl3dtruth_sppmax", &cl3dtruth_sppmax, &b_cl3dtruth_sppmax);
      fChain->SetBranchAddress("cl3dtruth_szz", &cl3dtruth_szz, &b_cl3dtruth_szz);
      fChain->SetBranchAddress("cl3dtruth_srrtot", &cl3dtruth_srrtot, &b_cl3dtruth_srrtot);
      fChain->SetBranchAddress("cl3dtruth_srrmax", &cl3dtruth_srrmax, &b_cl3dtruth_srrmax);
      fChain->SetBranchAddress("cl3dtruth_srrmean", &cl3dtruth_srrmean, &b_cl3dtruth_srrmean);
      fChain->SetBranchAddress("cl3dtruth_emaxe", &cl3dtruth_emaxe, &b_cl3dtruth_emaxe);
      fChain->SetBranchAddress("cl3dtruth_hoe", &cl3dtruth_hoe, &b_cl3dtruth_hoe);
      fChain->SetBranchAddress("cl3dtruth_meanz", &cl3dtruth_meanz, &b_cl3dtruth_meanz);
      fChain->SetBranchAddress("cl3dtruth_layer10", &cl3dtruth_layer10, &b_cl3dtruth_layer10);
      fChain->SetBranchAddress("cl3dtruth_layer50", &cl3dtruth_layer50, &b_cl3dtruth_layer50);
      fChain->SetBranchAddress("cl3dtruth_layer90", &cl3dtruth_layer90, &b_cl3dtruth_layer90);
      fChain->SetBranchAddress("cl3dtruth_ntc67", &cl3dtruth_ntc67, &b_cl3dtruth_ntc67);
      fChain->SetBranchAddress("cl3dtruth_ntc90", &cl3dtruth_ntc90, &b_cl3dtruth_ntc90);
      fChain->SetBranchAddress("cl3dtruth_bdteg", &cl3dtruth_bdteg, &b_cl3dtruth_bdteg);
      fChain->SetBranchAddress("cl3dtruth_quality", &cl3dtruth_quality, &b_cl3dtruth_quality);
      fChain->SetBranchAddress("cl3dtruth_ipt", &cl3dtruth_ipt, &b_cl3dtruth_ipt);
      fChain->SetBranchAddress("cl3dtruth_ienergy", &cl3dtruth_ienergy, &b_cl3dtruth_ienergy);
      fChain->SetBranchAddress("towertruth_n", &towertruth_n, &b_towertruth_n);
      fChain->SetBranchAddress("towertruth_pt", &towertruth_pt, &b_towertruth_pt);
      fChain->SetBranchAddress("towertruth_energy", &towertruth_energy, &b_towertruth_energy);
      fChain->SetBranchAddress("towertruth_eta", &towertruth_eta, &b_towertruth_eta);
      fChain->SetBranchAddress("towertruth_phi", &towertruth_phi, &b_towertruth_phi);
      fChain->SetBranchAddress("towertruth_etEm", &towertruth_etEm, &b_towertruth_etEm);
      fChain->SetBranchAddress("towertruth_etHad", &towertruth_etHad, &b_towertruth_etHad);
      fChain->SetBranchAddress("towertruth_iEta", &towertruth_iEta, &b_towertruth_iEta);
      fChain->SetBranchAddress("towertruth_iPhi", &towertruth_iPhi, &b_towertruth_iPhi);
   }
};

#endif
