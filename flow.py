import os
import re

import numpy as np

try:
    from sklearn.externals import joblib
except ImportError:
    import joblib


def execute(cmd, logger=None):
    if logger:
        logger.info("executing: %s " % cmd)
    else:
        print("[INFO]: executing: %s" % cmd)

    return os.system(cmd)
    
class modeling_flow():
    def __init__(
        self, configs
    ):
        self.model_path = configs["power-model-path"]
        self.template_xml = configs["xml-template-path"]
        self.gem5_stats = configs["gem5-stats-path"]
        self.gem5_config = configs["gem5-config-path"]
        self.output_path = configs["modleing-output-path"]
        self.mcpat_xml = os.path.join(
            self.output_path,
            "mcpat-input.xml"
        )
        self.parser_log = os.path.join(
            self.output_path,
            "gem5-mcpat-parser.log"
        )
        self.mcpat_report = os.path.join(
            self.output_path,
            "mcpat.rpt"
        )
        self.calibration_report = os.path.join(
            self.output_path,
            "ml-calibration.rpt"
        )

        self.dynamic_calib_features = np.zeros((1, 17))	
        self.leakage_calib_features = np.zeros((1, 2))

        self.leakage_model = joblib.load(
            os.path.join(self.model_path , "leakage-calib.pt")
        )	
        self.dynamic_model = joblib.load(
            os.path.join(self.model_path , "dynamic-calib.pt")
        )

    def gem5_to_mcpat(self):
        self.gem5_parser()
        self.mcpat_execute()


    def mcpat_to_calib(self):
        self.extract_mcpat_rpt()
        self.extract_gem5_stats()
        self.calibration()

    def gem5_parser(self):
        execute(
            "python2 %s -c %s -s %s -t %s -o %s > %s" % (
                os.path.join("parser", "gem5-mcpat-parser.py"),
                self.gem5_config,
                self.gem5_stats,
                self.template_xml,
                self.mcpat_xml,
                self.parser_log
            ),
        )

    def mcpat_execute(self):
        execute(
            "%s -infile %s -print_level 5 > %s" % (
                os.path.join("mcpat", "mcpat"),
                self.mcpat_xml,
                self.mcpat_report
            )
        )

    def extract_mcpat_rpt(self):
        p_subthreshold = re.compile(r"Subthreshold\ Leakage\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
        p_dynamic = re.compile(r"Runtime\ Dynamic\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ W")
        p_area = re.compile(r"Area\ =\ [+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\ mm\^2")
        with open(self.mcpat_report, 'r') as rpt:
            rpt = rpt.read()
            try:
                area = p_area.findall(rpt)
                core_area = float(area[1][0])
                leakage = p_subthreshold.findall(rpt)
                core_leakage = float(leakage[4][0])
                self.dynamic_calib_features[0][0] = core_leakage 
                self.leakage_calib_features[0][0] = core_leakage * 1000
                self.leakage_calib_features[0][1] = core_area
                dynamic = p_dynamic.findall(rpt)
                core_dynamic = float(dynamic[4][0])
                self.dynamic_calib_features[0][1] = core_dynamic 
                free_list_dynamic = float(dynamic[19][0])
                self.dynamic_calib_features[0][2] = free_list_dynamic 
                mmu_dynamic = float(dynamic[25][0] + dynamic[25][2])
                self.dynamic_calib_features[0][3] = mmu_dynamic 
                rob_dynamic = float(dynamic[35][0])
                self.dynamic_calib_features[0][4] = rob_dynamic 
            except Exception as e:
                print("[ERROR]:", e)
                exit(1)
        return core_leakage + core_dynamic

    def extract_gem5_stats(self):
        with open(self.gem5_stats, 'r') as f:
            for line in f.readlines():
                if "system.cpu.numCycles" in line:
                    numcycles = float(line.split()[1])

                if "system.cpu.iew.exec_stores" in line:
                    iew_exex_stores = float(line.split()[1])
                    
                if "system.cpu.iq.int_alu_accesses" in line:
                    int_alu_accesses = float(line.split()[1])
                    
                if "system.cpu.iq.FU_type_0::FloatMemRead" in line:
                    FU_FpMemRead = float(line.split()[1])
                    
                if "system.cpu.iq.FU_type_0::IntDiv" in line:
                    IntDiv = float(line.split()[1])
                    
                if "system.cpu.iq.FU_type_0::FloatMult" in line:
                    FU_FpMult = float(line.split()[1])
                    
                if "system.cpu.iq.FU_type_0::FloatDiv" in line:
                    FU_FpDiv = float(line.split()[1])
                     
                if "system.cpu.memDep0.conflictingStores" in line:
                    mem_conflictStores = float(line.split()[1])
                    
                if "system.cpu.rename.CommittedMaps" in line:
                    rename_Maps = float(line.split()[1])
                    
                if "system.mem_ctrls.readReqs" in line:
                    mem_ctrls_reads = float(line.split()[1])
                     
                if "system.cpu.icache.overall_mshr_hits::total" in line:
                    icache_mshr_hits = float(line.split()[1])
                     
                if "system.cpu.dcache.overall_accesses::total" in line:
                    dcache_accesses = float(line.split()[1])
                     
                if "system.cpu.dcache.overall_mshr_hits::total" in line:
                    dcache_mshr_hits = float(line.split()[1])
   
        self.dynamic_calib_features[0][5] = iew_exex_stores / numcycles 
        self.dynamic_calib_features[0][6] = int_alu_accesses / numcycles 
        self.dynamic_calib_features[0][7] = FU_FpMemRead / numcycles 
        self.dynamic_calib_features[0][8] = IntDiv / numcycles 
        self.dynamic_calib_features[0][9] = FU_FpMult / numcycles 
        self.dynamic_calib_features[0][10] = FU_FpDiv / numcycles
        self.dynamic_calib_features[0][11] = mem_conflictStores / numcycles 
        self.dynamic_calib_features[0][12] = rename_Maps / numcycles 
        self.dynamic_calib_features[0][13] = mem_ctrls_reads / numcycles
        self.dynamic_calib_features[0][14] = icache_mshr_hits / numcycles
        self.dynamic_calib_features[0][15] = dcache_accesses / numcycles
        self.dynamic_calib_features[0][16] = dcache_mshr_hits / numcycles                  


    def calibration(self):

        leakage_pred = self.leakage_model.predict(self.leakage_calib_features)[0]
        dynamic_pred = self.dynamic_model.predict(self.dynamic_calib_features)[0]
        total_pred = leakage_pred + dynamic_pred
        print("\n[INFO]: After Calibration")
        print("Results: Leakage = {:.3f} mW, Dynamic = {:.3f} mW, Total Power = {:.3f} mW".format(leakage_pred, dynamic_pred, total_pred))

        rpt = open(self.calibration_report,'w')
        rpt.write("Leakage Power  {:.3f}  mW\nDynamic Power  {:.3f}  mW\nTotal Power  {:.3f}  mW".format(leakage_pred, dynamic_pred, total_pred))
        rpt.close()
