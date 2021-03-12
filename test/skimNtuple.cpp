#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TString.h>
#include <TLorentzVector.h>
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"

using namespace std;


int main (int argc, char** argv)
{
    if (argc < 5)
    {
      cerr << "missing input parameters : argc is: " << argc << endl ;
      cerr << "usage: " << argv[0]
           << " inputFileNameList outputFileName nEvents isTau isQCD" << endl ;
      return 1;
    }

    int nEvents = atoi(argv[3]);
    int isTau = atoi(argv[4]);
    int isQCD = atoi(argv[5]);

    vector<TString> inFiles;
    TString file;
    ifstream inputFileNameList;
    inputFileNameList.open(argv[1], ios::in);
    while (!inputFileNameList.eof()){
        inputFileNameList >> file;
        if (file == "") break;
        inFiles.push_back(file);
    }

    TFile * out_file = TFile::Open(argv[2],"RECREATE");

    TChain * in_tree = new TChain("hgcalTriggerNtuplizer/HGCalTriggerNtuple");

    for (unsigned int ifile = 0; ifile<inFiles.size(); ifile++){
        in_tree->Add(inFiles[ifile]);
    }

    Long64_t nEntries = in_tree->GetEntries();
    cout<<"nEntries="<<in_tree->GetEntries()<<endl;
    
    if (nEvents != -1) nEntries = nEvents;

    // old branches used

    int _in_run;
    int _in_event;
    int _in_lumi;

    vector<float> *_in_gentau_pt; 
    vector<float> *_in_gentau_eta;
    vector<float> *_in_gentau_phi;
    vector<float> *_in_gentau_energy;
    vector<float> *_in_gentau_mass;

    vector<float> *_in_genjet_pt; 
    vector<float> *_in_genjet_eta;
    vector<float> *_in_genjet_phi;
    vector<float> *_in_genjet_energy;
    TLorentzVector in_genjet;

    vector<float> *_in_gentau_vis_pt; 
    vector<float> *_in_gentau_vis_eta;
    vector<float> *_in_gentau_vis_phi;
    vector<float> *_in_gentau_vis_energy;
    vector<float> *_in_gentau_vis_mass;

    vector<vector<float> > *_in_gentau_products_pt;
    vector<vector<float> > *_in_gentau_products_eta;
    vector<vector<float> > *_in_gentau_products_phi;
    vector<vector<float> > *_in_gentau_products_energy;
    vector<vector<float> > *_in_gentau_products_mass;
    vector<vector<float> > *_in_gentau_products_id;

    vector<int>   *_in_gentau_decayMode;
    vector<int>   *_in_gentau_totNproducts;
    vector<int>   *_in_gentau_totNgamma;
    vector<int>   *_in_gentau_totNpiZero;
    vector<int>   *_in_gentau_totNcharged;

    int _in_tc_n;

    vector<unsigned int>    *_in_tc_id;
    vector<int>     *_in_tc_subdet;
    vector<int>     *_in_tc_zside;
    vector<int>     *_in_tc_layer;
    vector<int>     *_in_tc_waferu;
    vector<int>     *_in_tc_waferv;
    vector<int>     *_in_tc_wafertype;
    vector<int>     *_in_tc_panel_number;
    vector<int>     *_in_tc_panel_sector;
    vector<int>     *_in_tc_cell;
    vector<int>     *_in_tc_cellu;
    vector<int>     *_in_tc_cellv;  
    vector<unsigned int>    *_in_tc_data;
    vector<unsigned int>    *_in_tc_uncompressedCharge;
    vector<unsigned int>    *_in_tc_compressedCharge;

    vector<float>   *_in_tc_pt;
    vector<float>   *_in_tc_mipPt;
    vector<float>   *_in_tc_energy;
    vector<float>   *_in_tc_eta;
    vector<float>   *_in_tc_phi;
    vector<float>   *_in_tc_x;
    vector<float>   *_in_tc_y;
    vector<float>   *_in_tc_z;

    vector<unsigned int>    *_in_tc_cluster_id;
    vector<unsigned int>    *_in_tc_multicluster_id;
    vector<unsigned int>    *_in_tc_multicluster_pt;

    int _in_cl3d_n;

    vector<unsigned int> *_in_cl3d_id;
    vector<float>        *_in_cl3d_pt;
    vector<float>        *_in_cl3d_energy;
    vector<float>        *_in_cl3d_eta;
    vector<float>        *_in_cl3d_phi;

    vector<int>                     *_in_cl3d_clusters_n;
    vector<vector<unsigned int> >   *_in_cl3d_clusters_id;

    vector<int>   *_in_cl3d_showerlength;
    vector<int>   *_in_cl3d_coreshowerlength;
    vector<int>   *_in_cl3d_firstlayer;
    vector<int>   *_in_cl3d_maxlayer;
    vector<float> *_in_cl3d_seetot;
    vector<float> *_in_cl3d_seemax;
    vector<float> *_in_cl3d_spptot;
    vector<float> *_in_cl3d_sppmax;
    vector<float> *_in_cl3d_szz;
    vector<float> *_in_cl3d_srrtot;
    vector<float> *_in_cl3d_srrmax;
    vector<float> *_in_cl3d_srrmean;
    vector<float> *_in_cl3d_emaxe;
    vector<float> *_in_cl3d_hoe;
    vector<float> *_in_cl3d_meanz;
    vector<float> *_in_cl3d_layer10;
    vector<float> *_in_cl3d_layer50;
    vector<float> *_in_cl3d_layer90;
    vector<float> *_in_cl3d_ntc67;
    vector<float> *_in_cl3d_ntc90;
    vector<float> *_in_cl3d_bdteg;
    vector<int>   *_in_cl3d_quality;

    in_tree->SetBranchAddress("run",    &_in_run);
    in_tree->SetBranchAddress("event",  &_in_event);
    in_tree->SetBranchAddress("lumi",   &_in_lumi);

    if(isTau){

        in_tree->SetBranchAddress("gentau_pt",      &_in_gentau_pt);
        in_tree->SetBranchAddress("gentau_eta",     &_in_gentau_eta);
        in_tree->SetBranchAddress("gentau_phi",     &_in_gentau_phi);
        in_tree->SetBranchAddress("gentau_energy",  &_in_gentau_energy);
        in_tree->SetBranchAddress("gentau_mass",    &_in_gentau_mass);

        in_tree->SetBranchAddress("gentau_vis_pt",      &_in_gentau_vis_pt);
        in_tree->SetBranchAddress("gentau_vis_eta",     &_in_gentau_vis_eta);
        in_tree->SetBranchAddress("gentau_vis_phi",     &_in_gentau_vis_phi);
        in_tree->SetBranchAddress("gentau_vis_energy",  &_in_gentau_vis_energy);
        in_tree->SetBranchAddress("gentau_vis_mass",    &_in_gentau_vis_mass);

        in_tree->SetBranchAddress("gentau_products_pt",     &_in_gentau_products_pt);
        in_tree->SetBranchAddress("gentau_products_eta",    &_in_gentau_products_eta);
        in_tree->SetBranchAddress("gentau_products_phi",    &_in_gentau_products_phi);
        in_tree->SetBranchAddress("gentau_products_energy", &_in_gentau_products_energy);
        in_tree->SetBranchAddress("gentau_products_mass",   &_in_gentau_products_mass);
        in_tree->SetBranchAddress("gentau_products_id",     &_in_gentau_products_id);

        in_tree->SetBranchAddress("gentau_decayMode",       &_in_gentau_decayMode);
        in_tree->SetBranchAddress("gentau_totNproducts",    &_in_gentau_totNproducts);
        in_tree->SetBranchAddress("gentau_totNgamma",       &_in_gentau_totNgamma);
        in_tree->SetBranchAddress("gentau_totNpiZero",      &_in_gentau_totNpiZero);
        in_tree->SetBranchAddress("gentau_totNcharged",     &_in_gentau_totNcharged);

    }
    else if(isQCD){

        in_tree->SetBranchAddress("genjet_pt",      &_in_genjet_pt);
        in_tree->SetBranchAddress("genjet_eta",     &_in_genjet_eta);
        in_tree->SetBranchAddress("genjet_phi",     &_in_genjet_phi);
        in_tree->SetBranchAddress("genjet_energy",  &_in_genjet_energy);

    }

    in_tree->SetBranchAddress("tc_n",   &_in_tc_n);

    in_tree->SetBranchAddress("tc_id",      &_in_tc_id);
    in_tree->SetBranchAddress("tc_subdet",  &_in_tc_subdet);
    in_tree->SetBranchAddress("tc_zside",   &_in_tc_zside);
    in_tree->SetBranchAddress("tc_layer",   &_in_tc_layer);
    in_tree->SetBranchAddress("tc_waferu",  &_in_tc_waferu);
    in_tree->SetBranchAddress("tc_waferv",  &_in_tc_waferv);
    in_tree->SetBranchAddress("tc_wafertype",   &_in_tc_wafertype);
    in_tree->SetBranchAddress("tc_panel_number",    &_in_tc_panel_number);
    in_tree->SetBranchAddress("tc_panel_sector",    &_in_tc_panel_sector);
    in_tree->SetBranchAddress("tc_cellu",           &_in_tc_cellu);
    in_tree->SetBranchAddress("tc_cellv",           &_in_tc_cellv); 
    in_tree->SetBranchAddress("tc_data",            &_in_tc_data);
    in_tree->SetBranchAddress("tc_uncompressedCharge",  &_in_tc_uncompressedCharge);
    in_tree->SetBranchAddress("tc_compressedCharge",    &_in_tc_compressedCharge);

    in_tree->SetBranchAddress("tc_pt",              &_in_tc_pt);
    in_tree->SetBranchAddress("tc_mipPt",           &_in_tc_mipPt);
    in_tree->SetBranchAddress("tc_energy",          &_in_tc_energy);
    in_tree->SetBranchAddress("tc_eta",             &_in_tc_eta);
    in_tree->SetBranchAddress("tc_phi",             &_in_tc_phi);
    in_tree->SetBranchAddress("tc_x",               &_in_tc_x);
    in_tree->SetBranchAddress("tc_y",               &_in_tc_y);
    in_tree->SetBranchAddress("tc_z",               &_in_tc_z);

    in_tree->SetBranchAddress("tc_cluster_id",      &_in_tc_cluster_id);
    in_tree->SetBranchAddress("tc_multicluster_id", &_in_tc_multicluster_id);
    in_tree->SetBranchAddress("tc_multicluster_pt", &_in_tc_multicluster_pt);

    in_tree->SetBranchAddress("cl3d_n",&_in_cl3d_n);

    in_tree->SetBranchAddress("cl3d_id",        &_in_cl3d_id);
    in_tree->SetBranchAddress("cl3d_pt",        &_in_cl3d_pt);
    in_tree->SetBranchAddress("cl3d_energy",    &_in_cl3d_energy);
    in_tree->SetBranchAddress("cl3d_eta",       &_in_cl3d_eta);
    in_tree->SetBranchAddress("cl3d_phi",       &_in_cl3d_phi);

    in_tree->SetBranchAddress("cl3d_clusters_n",    &_in_cl3d_clusters_n);
    in_tree->SetBranchAddress("cl3d_clusters_id",   &_in_cl3d_clusters_id);

    in_tree->SetBranchAddress("cl3d_showerlength",      &_in_cl3d_showerlength);
    in_tree->SetBranchAddress("cl3d_coreshowerlength",  &_in_cl3d_coreshowerlength);
    in_tree->SetBranchAddress("cl3d_firstlayer",        &_in_cl3d_firstlayer);
    in_tree->SetBranchAddress("cl3d_maxlayer",          &_in_cl3d_maxlayer);
    in_tree->SetBranchAddress("cl3d_seetot",    &_in_cl3d_seetot);
    in_tree->SetBranchAddress("cl3d_seemax",    &_in_cl3d_seemax);
    in_tree->SetBranchAddress("cl3d_spptot",    &_in_cl3d_spptot);
    in_tree->SetBranchAddress("cl3d_sppmax",    &_in_cl3d_sppmax);
    in_tree->SetBranchAddress("cl3d_szz",       &_in_cl3d_szz);
    in_tree->SetBranchAddress("cl3d_srrtot",    &_in_cl3d_srrtot);
    in_tree->SetBranchAddress("cl3d_srrmax",    &_in_cl3d_srrmax);
    in_tree->SetBranchAddress("cl3d_srrmean",   &_in_cl3d_srrmean);
    in_tree->SetBranchAddress("cl3d_emaxe",     &_in_cl3d_emaxe);
    in_tree->SetBranchAddress("cl3d_hoe",       &_in_cl3d_hoe);
    in_tree->SetBranchAddress("cl3d_meanz",     &_in_cl3d_meanz);
    in_tree->SetBranchAddress("cl3d_layer10",   &_in_cl3d_layer10);
    in_tree->SetBranchAddress("cl3d_layer50",   &_in_cl3d_layer50);
    in_tree->SetBranchAddress("cl3d_layer90",   &_in_cl3d_layer90);
    in_tree->SetBranchAddress("cl3d_ntc67",     &_in_cl3d_ntc67);
    in_tree->SetBranchAddress("cl3d_ntc90",     &_in_cl3d_ntc90);
    in_tree->SetBranchAddress("cl3d_bdteg",     &_in_cl3d_bdteg);
    in_tree->SetBranchAddress("cl3d_quality",   &_in_cl3d_quality);


    TTree* out_tree = new TTree("SkimmedTree","SkimmedTree");

    int _out_run;
    int _out_event;
    int _out_lumi;

    int _out_gentau_n;
    int _out_genjet_n;

    vector<float> _out_gentau_pt; 
    vector<float> _out_gentau_eta;
    vector<float> _out_gentau_phi;
    vector<float> _out_gentau_energy;
    vector<float> _out_gentau_mass;

    vector<float> _out_genjet_pt; 
    vector<float> _out_genjet_eta;
    vector<float> _out_genjet_phi;
    vector<float> _out_genjet_energy;
    vector<float> _out_genjet_mass;

    vector<float> _out_gentau_vis_pt; 
    vector<float> _out_gentau_vis_eta;
    vector<float> _out_gentau_vis_phi;
    vector<float> _out_gentau_vis_energy;
    vector<float> _out_gentau_vis_mass;

    vector<vector<float> > _out_gentau_products_pt;
    vector<vector<float> > _out_gentau_products_eta;
    vector<vector<float> > _out_gentau_products_phi;
    vector<vector<float> > _out_gentau_products_energy;
    vector<vector<float> > _out_gentau_products_mass;
    vector<vector<float> > _out_gentau_products_id;

    vector<int>   _out_gentau_decayMode;
    vector<int>   _out_gentau_totNproducts;
    vector<int>   _out_gentau_totNgamma;
    vector<int>   _out_gentau_totNpiZero;
    vector<int>   _out_gentau_totNcharged;

    int _out_tc_n;

    vector<unsigned int>    _out_tc_id;
    vector<int>     _out_tc_subdet;
    vector<int>     _out_tc_zside;
    vector<int>     _out_tc_layer;
    vector<int>     _out_tc_waferu;
    vector<int>     _out_tc_waferv;
    vector<int>     _out_tc_wafertype;
    vector<int>     _out_tc_panel_number;
    vector<int>     _out_tc_panel_sector;
    vector<int>     _out_tc_cellu;
    vector<int>     _out_tc_cellv;  
    vector<unsigned int>    _out_tc_data;
    vector<unsigned int>    _out_tc_uncompressedCharge;
    vector<unsigned int>    _out_tc_compressedCharge;

    vector<float>   _out_tc_pt;
    vector<float>   _out_tc_mipPt;
    vector<float>   _out_tc_energy;
    vector<float>   _out_tc_eta;
    vector<float>   _out_tc_phi;
    vector<float>   _out_tc_x;
    vector<float>   _out_tc_y;
    vector<float>   _out_tc_z;

    vector<unsigned int>    _out_tc_cluster_id;
    vector<unsigned int>    _out_tc_multicluster_id;
    vector<unsigned int>    _out_tc_multicluster_pt;

    int _out_cl3d_n;

    vector<unsigned int> _out_cl3d_id;
    vector<float>        _out_cl3d_pt;
    vector<float>        _out_cl3d_energy;
    vector<float>        _out_cl3d_eta;
    vector<float>        _out_cl3d_phi;

    vector<int>                     _out_cl3d_clusters_n;
    vector<vector<unsigned int> >   _out_cl3d_clusters_id;

    vector<int>   _out_cl3d_showerlength;
    vector<int>   _out_cl3d_coreshowerlength;
    vector<int>   _out_cl3d_firstlayer;
    vector<int>   _out_cl3d_maxlayer;
    vector<float> _out_cl3d_seetot;
    vector<float> _out_cl3d_seemax;
    vector<float> _out_cl3d_spptot;
    vector<float> _out_cl3d_sppmax;
    vector<float> _out_cl3d_szz;
    vector<float> _out_cl3d_srrtot;
    vector<float> _out_cl3d_srrmax;
    vector<float> _out_cl3d_srrmean;
    vector<float> _out_cl3d_emaxe;
    vector<float> _out_cl3d_hoe;
    vector<float> _out_cl3d_meanz;
    vector<float> _out_cl3d_layer10;
    vector<float> _out_cl3d_layer50;
    vector<float> _out_cl3d_layer90;
    vector<float> _out_cl3d_ntc67;
    vector<float> _out_cl3d_ntc90;
    vector<float> _out_cl3d_bdteg;
    vector<int>   _out_cl3d_quality;

    out_tree->Branch("run",     &_in_run);
    out_tree->Branch("event",   &_in_event);
    out_tree->Branch("lumi",    &_in_lumi);

    if(isTau){

        out_tree->Branch("gentau_n",&_out_gentau_n);

        out_tree->Branch("gentau_pt",       &_out_gentau_pt);
        out_tree->Branch("gentau_eta",      &_out_gentau_eta);
        out_tree->Branch("gentau_phi",      &_out_gentau_phi);
        out_tree->Branch("gentau_energy",   &_out_gentau_energy);
        out_tree->Branch("gentau_mass",     &_out_gentau_mass);

        out_tree->Branch("gentau_vis_pt",       &_out_gentau_vis_pt);
        out_tree->Branch("gentau_vis_eta",      &_out_gentau_vis_eta);
        out_tree->Branch("gentau_vis_phi",      &_out_gentau_vis_phi);
        out_tree->Branch("gentau_vis_energy",   &_out_gentau_vis_energy);
        out_tree->Branch("gentau_vis_mass",     &_out_gentau_vis_mass);

        out_tree->Branch("gentau_products_pt",      &_out_gentau_products_pt);
        out_tree->Branch("gentau_products_eta",     &_out_gentau_products_eta);
        out_tree->Branch("gentau_products_phi",     &_out_gentau_products_phi);
        out_tree->Branch("gentau_products_energy",  &_out_gentau_products_energy);
        out_tree->Branch("gentau_products_mass",    &_out_gentau_products_mass);
        out_tree->Branch("gentau_products_id",      &_out_gentau_products_id);

        out_tree->Branch("gentau_decayMode",        &_out_gentau_decayMode);
        out_tree->Branch("gentau_totNproducts",     &_out_gentau_totNproducts);
        out_tree->Branch("gentau_totNgamma",        &_out_gentau_totNgamma);
        out_tree->Branch("gentau_totNpiZero",       &_out_gentau_totNpiZero);
        out_tree->Branch("gentau_totNcharged",      &_out_gentau_totNcharged);

    }
    else if(isQCD){

        out_tree->Branch("genjet_n",&_out_genjet_n);

        out_tree->Branch("genjet_pt",       &_out_genjet_pt);
        out_tree->Branch("genjet_eta",      &_out_genjet_eta);
        out_tree->Branch("genjet_phi",      &_out_genjet_phi);
        out_tree->Branch("genjet_energy",   &_out_genjet_energy);
        out_tree->Branch("genjet_mass",     &_out_genjet_mass);

    }

    out_tree->Branch("tc_n",                &_out_tc_n);

    out_tree->Branch("tc_id",               &_out_tc_id);
    out_tree->Branch("tc_subdet",           &_out_tc_subdet);
    out_tree->Branch("tc_zside",            &_out_tc_zside);
    out_tree->Branch("tc_layer",            &_out_tc_layer);
    out_tree->Branch("tc_waferu",           &_out_tc_waferu);
    out_tree->Branch("tc_waferv",           &_out_tc_waferv);
    out_tree->Branch("tc_wafertype",        &_out_tc_wafertype);
    out_tree->Branch("tc_panel_number",     &_out_tc_panel_number);
    out_tree->Branch("tc_panel_sector",     &_out_tc_panel_sector);
    out_tree->Branch("tc_cellu",            &_out_tc_cellu);
    out_tree->Branch("tc_cellv",            &_out_tc_cellv);    
    out_tree->Branch("tc_data",                 &_out_tc_data);
    out_tree->Branch("tc_uncompressedCharge",   &_out_tc_uncompressedCharge);
    out_tree->Branch("tc_compressedCharge",     &_out_tc_compressedCharge);

    out_tree->Branch("tc_pt",               &_out_tc_pt);
    out_tree->Branch("tc_mipPt",            &_out_tc_mipPt);
    out_tree->Branch("tc_energy",           &_out_tc_energy);
    out_tree->Branch("tc_eta",              &_out_tc_eta);
    out_tree->Branch("tc_phi",              &_out_tc_phi);
    out_tree->Branch("tc_x",                &_out_tc_x);
    out_tree->Branch("tc_y",                &_out_tc_y);
    out_tree->Branch("tc_z",                &_out_tc_z);

    out_tree->Branch("tc_cluster_id",       &_out_tc_cluster_id);
    out_tree->Branch("tc_multicluster_id",  &_out_tc_multicluster_id);
    out_tree->Branch("tc_multicluster_pt",  &_out_tc_multicluster_pt);

    out_tree->Branch("cl3d_n",&_out_cl3d_n);

    out_tree->Branch("cl3d_id",     &_out_cl3d_id);
    out_tree->Branch("cl3d_pt",     &_out_cl3d_pt);
    out_tree->Branch("cl3d_energy", &_out_cl3d_energy);
    out_tree->Branch("cl3d_eta",    &_out_cl3d_eta);
    out_tree->Branch("cl3d_phi",    &_out_cl3d_phi);

    out_tree->Branch("cl3d_clusters_n", &_out_cl3d_clusters_n);
    out_tree->Branch("cl3d_clusters_id",    &_out_cl3d_clusters_id);

    out_tree->Branch("cl3d_showerlength",       &_out_cl3d_showerlength);
    out_tree->Branch("cl3d_coreshowerlength",   &_out_cl3d_coreshowerlength);
    out_tree->Branch("cl3d_firstlayer",         &_out_cl3d_firstlayer);
    out_tree->Branch("cl3d_maxlayer",           &_out_cl3d_maxlayer);
    out_tree->Branch("cl3d_seetot",     &_out_cl3d_seetot);
    out_tree->Branch("cl3d_seemax",     &_out_cl3d_seemax);
    out_tree->Branch("cl3d_spptot",     &_out_cl3d_spptot);
    out_tree->Branch("cl3d_sppmax",     &_out_cl3d_sppmax);
    out_tree->Branch("cl3d_szz",        &_out_cl3d_szz);
    out_tree->Branch("cl3d_srrtot",     &_out_cl3d_srrtot);
    out_tree->Branch("cl3d_srrmax",     &_out_cl3d_srrmax);
    out_tree->Branch("cl3d_srrmean",    &_out_cl3d_srrmean);
    out_tree->Branch("cl3d_emaxe",      &_out_cl3d_emaxe);
    out_tree->Branch("cl3d_hoe",        &_out_cl3d_hoe);
    out_tree->Branch("cl3d_meanz",      &_out_cl3d_meanz);
    out_tree->Branch("cl3d_layer10",    &_out_cl3d_layer10);
    out_tree->Branch("cl3d_layer50",    &_out_cl3d_layer50);
    out_tree->Branch("cl3d_layer90",    &_out_cl3d_layer90);
    out_tree->Branch("cl3d_ntc67",      &_out_cl3d_ntc67);
    out_tree->Branch("cl3d_ntc90",      &_out_cl3d_ntc90);
    out_tree->Branch("cl3d_bdteg",      &_out_cl3d_bdteg);
    out_tree->Branch("cl3d_quality",    &_out_cl3d_quality);


    for (int i=0;i<nEntries;i++) {

        if(i%1000==0) cout<<"i="<<i<<endl;

        //old branches

        _in_gentau_pt = 0; 
        _in_gentau_eta = 0;
        _in_gentau_phi = 0;
        _in_gentau_energy = 0;
        _in_gentau_mass = 0;

        _in_genjet_pt = 0; 
        _in_genjet_eta = 0;
        _in_genjet_phi = 0;
        _in_genjet_energy = 0;
        
        _in_gentau_vis_pt = 0; 
        _in_gentau_vis_eta = 0;
        _in_gentau_vis_phi = 0;
        _in_gentau_vis_energy = 0;
        _in_gentau_vis_mass = 0;
        
        _in_gentau_products_pt = 0;
        _in_gentau_products_eta = 0;
        _in_gentau_products_phi = 0;
        _in_gentau_products_energy = 0;
        _in_gentau_products_mass = 0;
        _in_gentau_products_id = 0;
        
        _in_gentau_decayMode = 0;
        _in_gentau_totNproducts = 0;
        _in_gentau_totNgamma = 0;
        _in_gentau_totNpiZero = 0;
        _in_gentau_totNcharged = 0;

        _in_tc_n = 0;

        _in_tc_id = 0;
        _in_tc_subdet = 0;
        _in_tc_zside = 0;
        _in_tc_layer = 0;
        _in_tc_waferu = 0;
        _in_tc_waferv = 0;
        _in_tc_wafertype = 0;
        _in_tc_panel_number = 0;
        _in_tc_panel_sector = 0;
        _in_tc_cellu = 0;
        _in_tc_cellv = 0;
        _in_tc_data = 0;
        _in_tc_uncompressedCharge = 0;
        _in_tc_compressedCharge = 0;

        _in_tc_pt = 0;
        _in_tc_mipPt = 0;
        _in_tc_energy = 0;
        _in_tc_eta = 0;
        _in_tc_phi = 0;
        _in_tc_x = 0;
        _in_tc_y = 0;
        _in_tc_z = 0;

        _in_tc_cluster_id = 0;
        _in_tc_multicluster_id = 0;
        _in_tc_multicluster_pt = 0;

        _in_cl3d_n = 0;
        
        _in_cl3d_id = 0;
        _in_cl3d_pt = 0;
        _in_cl3d_energy = 0;
        _in_cl3d_eta = 0;
        _in_cl3d_phi = 0;
        
        _in_cl3d_clusters_n = 0;
        _in_cl3d_clusters_id = 0;
        
        _in_cl3d_showerlength = 0;
        _in_cl3d_coreshowerlength = 0;
        _in_cl3d_firstlayer = 0;
        _in_cl3d_maxlayer = 0;      
        _in_cl3d_seetot = 0;
        _in_cl3d_seemax = 0;
        _in_cl3d_spptot = 0;
        _in_cl3d_sppmax = 0;
        _in_cl3d_szz = 0;
        _in_cl3d_srrtot = 0;
        _in_cl3d_srrmax = 0;
        _in_cl3d_srrmean = 0;
        _in_cl3d_emaxe = 0;
        _in_cl3d_hoe = 0;
        _in_cl3d_meanz = 0;
        _in_cl3d_layer10 = 0;
        _in_cl3d_layer50 = 0;
        _in_cl3d_layer90 = 0;
        _in_cl3d_ntc67 = 0;
        _in_cl3d_ntc90 = 0;
        _in_cl3d_bdteg = 0;
        _in_cl3d_quality = 0;

        //new branches

        _out_gentau_n = 0;

        _out_gentau_pt.clear(); 
        _out_gentau_eta.clear();
        _out_gentau_phi.clear();
        _out_gentau_energy.clear();
        _out_gentau_mass.clear();

        _out_genjet_pt.clear(); 
        _out_genjet_eta.clear();
        _out_genjet_phi.clear();
        _out_genjet_energy.clear();
        _out_genjet_mass.clear();
        
        _out_gentau_vis_pt.clear(); 
        _out_gentau_vis_eta.clear();
        _out_gentau_vis_phi.clear();
        _out_gentau_vis_energy.clear();
        _out_gentau_vis_mass.clear();
        
        _out_gentau_products_pt.clear();
        _out_gentau_products_eta.clear();
        _out_gentau_products_phi.clear();
        _out_gentau_products_energy.clear();
        _out_gentau_products_mass.clear();
        _out_gentau_products_id.clear();
        
        _out_gentau_decayMode.clear();
        _out_gentau_totNproducts.clear();
        _out_gentau_totNgamma.clear();
        _out_gentau_totNpiZero.clear();
        _out_gentau_totNcharged.clear();

        _out_tc_n = 0;

        _out_tc_id.clear();
        _out_tc_subdet.clear();
        _out_tc_zside.clear();
        _out_tc_layer.clear();
        _out_tc_waferu.clear();
        _out_tc_waferv.clear();
        _out_tc_wafertype.clear();
        _out_tc_panel_number.clear();
        _out_tc_panel_sector.clear();
        _out_tc_cellu.clear();
        _out_tc_cellv.clear();
        _out_tc_data.clear();
        _out_tc_uncompressedCharge.clear();
        _out_tc_compressedCharge.clear();

        _out_tc_pt.clear();
        _out_tc_mipPt.clear();
        _out_tc_energy.clear();
        _out_tc_eta.clear();
        _out_tc_phi.clear();
        _out_tc_x.clear();
        _out_tc_y.clear();
        _out_tc_z.clear();

        _out_tc_cluster_id.clear();
        _out_tc_multicluster_id.clear();
        _out_tc_multicluster_pt.clear();

        _out_cl3d_n = 0;
        
        _out_cl3d_id.clear();
        _out_cl3d_pt.clear();
        _out_cl3d_energy.clear();
        _out_cl3d_eta.clear();
        _out_cl3d_phi.clear();
        
        _out_cl3d_clusters_n.clear();
        _out_cl3d_clusters_id.clear();
        
        _out_cl3d_showerlength.clear();
        _out_cl3d_coreshowerlength.clear();
        _out_cl3d_firstlayer.clear();
        _out_cl3d_maxlayer.clear();
        _out_cl3d_seetot.clear();
        _out_cl3d_seemax.clear();
        _out_cl3d_spptot.clear();
        _out_cl3d_sppmax.clear();
        _out_cl3d_szz.clear();
        _out_cl3d_srrtot.clear();
        _out_cl3d_srrmax.clear();
        _out_cl3d_srrmean.clear();
        _out_cl3d_emaxe.clear();
        _out_cl3d_hoe.clear();
        _out_cl3d_meanz.clear();
        _out_cl3d_layer10.clear();
        _out_cl3d_layer50.clear();
        _out_cl3d_layer90.clear();
        _out_cl3d_ntc67.clear();
        _out_cl3d_ntc90.clear();
        _out_cl3d_bdteg.clear();
        _out_cl3d_quality.clear();

        //loop through entries

        int entry_ok = in_tree->GetEntry(i);
        if(entry_ok<0) continue;

        _out_run   = _in_run;
        _out_event = _in_event;
        _out_lumi  = _in_lumi;

        // loop over gentaus

        if(isTau){

            int n_gentaus = (*_in_gentau_pt).size();

            for (int i_gentau=0; i_gentau<n_gentaus; i_gentau++){

                if ( abs( (*_in_gentau_eta)[i_gentau] ) <= 1.5 || abs( (*_in_gentau_eta)[i_gentau] ) >= 3.0 ) continue;

                bool ishadronic = ( ((*_in_gentau_decayMode)[i_gentau] == 0) || ((*_in_gentau_decayMode)[i_gentau] == 1) || ((*_in_gentau_decayMode)[i_gentau] == 5) || ((*_in_gentau_decayMode)[i_gentau] == 10) || ((*_in_gentau_decayMode)[i_gentau] == 11) );

                if(!ishadronic) continue;

                _out_gentau_pt.push_back((*_in_gentau_pt)[i_gentau]);
                _out_gentau_eta.push_back((*_in_gentau_eta)[i_gentau]);
                _out_gentau_phi.push_back((*_in_gentau_phi)[i_gentau]);
                _out_gentau_energy.push_back((*_in_gentau_energy)[i_gentau]);
                _out_gentau_mass.push_back((*_in_gentau_mass)[i_gentau]);
        
                _out_gentau_vis_pt.push_back((*_in_gentau_vis_pt)[i_gentau]);
                _out_gentau_vis_eta.push_back((*_in_gentau_vis_eta)[i_gentau]);
                _out_gentau_vis_phi.push_back((*_in_gentau_vis_phi)[i_gentau]);
                _out_gentau_vis_energy.push_back((*_in_gentau_vis_energy)[i_gentau]);
                _out_gentau_vis_mass.push_back((*_in_gentau_vis_mass)[i_gentau]);
        
                _out_gentau_products_pt.push_back((*_in_gentau_products_pt)[i_gentau]);
                _out_gentau_products_eta.push_back((*_in_gentau_products_eta)[i_gentau]);
                _out_gentau_products_phi.push_back((*_in_gentau_products_phi)[i_gentau]);
                _out_gentau_products_energy.push_back((*_in_gentau_products_energy)[i_gentau]);
                _out_gentau_products_mass.push_back((*_in_gentau_products_mass)[i_gentau]);
                _out_gentau_products_id.push_back((*_in_gentau_products_id)[i_gentau]);
        
                _out_gentau_decayMode.push_back((*_in_gentau_decayMode)[i_gentau]);
                _out_gentau_totNproducts.push_back((*_in_gentau_totNproducts)[i_gentau]);
                _out_gentau_totNgamma.push_back((*_in_gentau_totNgamma)[i_gentau]);
                _out_gentau_totNpiZero.push_back((*_in_gentau_totNpiZero)[i_gentau]);
                _out_gentau_totNcharged.push_back((*_in_gentau_totNcharged)[i_gentau]);         

            }

            _out_gentau_n = _out_gentau_pt.size();

        }

        else if(isQCD){ 

            int n_genjets = (*_in_genjet_pt).size();

            for (int i_genjet=0; i_genjet<n_genjets; i_genjet++){

                if ( abs( (*_in_genjet_eta)[i_genjet] ) <= 1.5 || abs( (*_in_genjet_eta)[i_genjet] ) >= 3.0 ) continue;
                if ( (*_in_genjet_pt)[i_genjet] < 15 || (*_in_genjet_pt)[i_genjet] > 500 ) continue;

                _out_genjet_pt.push_back((*_in_genjet_pt)[i_genjet]);
                _out_genjet_eta.push_back((*_in_genjet_eta)[i_genjet]);
                _out_genjet_phi.push_back((*_in_genjet_phi)[i_genjet]);
                _out_genjet_energy.push_back((*_in_genjet_energy)[i_genjet]);
                in_genjet.SetPtEtaPhiE((*_in_genjet_pt)[i_genjet], (*_in_genjet_eta)[i_genjet], (*_in_genjet_phi)[i_genjet], (*_in_genjet_energy)[i_genjet]);
                _out_genjet_mass.push_back(in_genjet.M()); 

            }

            _out_genjet_n = _out_genjet_pt.size();

        }


        // loop over trigger cells

        _out_tc_n = _in_tc_n;

        for (int i_tc=0; i_tc<_in_tc_n; i_tc++){

            if ( abs( (*_in_tc_eta)[i_tc] ) <= 1.5 || abs( (*_in_tc_eta)[i_tc] ) >= 3.0 ) continue;

            _out_tc_id.push_back((*_in_tc_id)[i_tc]);
            _out_tc_subdet.push_back((*_in_tc_subdet)[i_tc]);
            _out_tc_zside.push_back((*_in_tc_zside)[i_tc]);
            _out_tc_layer.push_back((*_in_tc_layer)[i_tc]);
            _out_tc_waferu.push_back((*_in_tc_waferu)[i_tc]);
            _out_tc_waferv.push_back((*_in_tc_waferv)[i_tc]);
            _out_tc_wafertype.push_back((*_in_tc_wafertype)[i_tc]);
            _out_tc_panel_number.push_back((*_in_tc_panel_number)[i_tc]);
            _out_tc_panel_sector.push_back((*_in_tc_panel_sector)[i_tc]);
            _out_tc_cellu.push_back((*_in_tc_cellu)[i_tc]);
            _out_tc_cellv.push_back((*_in_tc_cellv)[i_tc]);
            _out_tc_data.push_back((*_in_tc_data)[i_tc]);
            _out_tc_uncompressedCharge.push_back((*_in_tc_uncompressedCharge)[i_tc]);
            _out_tc_compressedCharge.push_back((*_in_tc_compressedCharge)[i_tc]);

            _out_tc_pt.push_back((*_in_tc_pt)[i_tc]);
            _out_tc_mipPt.push_back((*_in_tc_mipPt)[i_tc]);
            _out_tc_energy.push_back((*_in_tc_energy)[i_tc]);
            _out_tc_eta.push_back((*_in_tc_eta)[i_tc]);
            _out_tc_phi.push_back((*_in_tc_phi)[i_tc]);
            _out_tc_x.push_back((*_in_tc_x)[i_tc]);
            _out_tc_y.push_back((*_in_tc_y)[i_tc]);
            _out_tc_z.push_back((*_in_tc_z)[i_tc]);

            _out_tc_cluster_id.push_back((*_in_tc_cluster_id)[i_tc]);
            _out_tc_multicluster_id.push_back((*_in_tc_multicluster_id)[i_tc]);
            _out_tc_multicluster_pt.push_back((*_in_tc_multicluster_pt)[i_tc]);

        }

        // loop over 3d clusters

        _out_cl3d_n = _in_cl3d_n;

        for (int i_cl3d=0; i_cl3d<_in_cl3d_n; i_cl3d++){    

            if ( abs( (*_in_cl3d_eta)[i_cl3d] ) <= 1.5 || abs( (*_in_cl3d_eta)[i_cl3d] ) >= 3.0 ) continue;

            _out_cl3d_id.push_back((*_in_cl3d_id)[i_cl3d]);
            _out_cl3d_pt.push_back((*_in_cl3d_pt)[i_cl3d]);
            _out_cl3d_energy.push_back((*_in_cl3d_energy)[i_cl3d]);
            _out_cl3d_eta.push_back((*_in_cl3d_eta)[i_cl3d]);
            _out_cl3d_phi.push_back((*_in_cl3d_phi)[i_cl3d]);
        
            _out_cl3d_clusters_n.push_back((*_in_cl3d_clusters_n)[i_cl3d]);
            _out_cl3d_clusters_id.push_back((*_in_cl3d_clusters_id)[i_cl3d]);
        
            _out_cl3d_showerlength.push_back((*_in_cl3d_showerlength)[i_cl3d]);
            _out_cl3d_coreshowerlength.push_back((*_in_cl3d_coreshowerlength)[i_cl3d]);
            _out_cl3d_firstlayer.push_back((*_in_cl3d_firstlayer)[i_cl3d]);
            _out_cl3d_maxlayer.push_back((*_in_cl3d_maxlayer)[i_cl3d]);     
            _out_cl3d_seetot.push_back((*_in_cl3d_seetot)[i_cl3d]);
            _out_cl3d_seemax.push_back((*_in_cl3d_seemax)[i_cl3d]);
            _out_cl3d_spptot.push_back((*_in_cl3d_spptot)[i_cl3d]);
            _out_cl3d_sppmax.push_back((*_in_cl3d_sppmax)[i_cl3d]);
            _out_cl3d_szz.push_back((*_in_cl3d_szz)[i_cl3d]);
            _out_cl3d_srrtot.push_back((*_in_cl3d_srrtot)[i_cl3d]);
            _out_cl3d_srrmax.push_back((*_in_cl3d_srrmax)[i_cl3d]);
            _out_cl3d_srrmean.push_back((*_in_cl3d_srrmean)[i_cl3d]);
            _out_cl3d_emaxe.push_back((*_in_cl3d_emaxe)[i_cl3d]);
            _out_cl3d_hoe.push_back((*_in_cl3d_hoe)[i_cl3d]);
            _out_cl3d_meanz.push_back((*_in_cl3d_meanz)[i_cl3d]);
            _out_cl3d_layer10.push_back((*_in_cl3d_layer10)[i_cl3d]);
            _out_cl3d_layer50.push_back((*_in_cl3d_layer50)[i_cl3d]);
            _out_cl3d_layer90.push_back((*_in_cl3d_layer90)[i_cl3d]);
            _out_cl3d_ntc67.push_back((*_in_cl3d_ntc67)[i_cl3d]);
            _out_cl3d_ntc90.push_back((*_in_cl3d_ntc90)[i_cl3d]);
            _out_cl3d_bdteg.push_back((*_in_cl3d_bdteg)[i_cl3d]);
            _out_cl3d_quality.push_back((*_in_cl3d_quality)[i_cl3d]);

        }

        out_tree->Fill();

    }

    out_file->cd();
    out_tree->Write();
    out_file->Close();

    cout << "... SKIM finished, exiting." << endl;

    return 0;
}