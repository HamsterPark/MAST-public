"""Built-in citation database for all known tools, algorithms, and papers.

Maps skill names and function names to their Citation entries.
A single skill can have multiple citations (algorithm paper + software).
"""

from __future__ import annotations

from mast.core.types import Citation, CitationType

# ─── Infrastructure (always cited if any Nanonis call was made) ────────────

MAST = Citation(
    key="MAST_2026",
    authors="",  # TODO: add authors
    title="MAST: Modular Autonomous SPM Toolkit for LLM-Driven Scanning Tunneling Microscopy",
    year=2026,
    citation_type=CitationType.SOFTWARE,
    url="",
    note="Instrument control framework",
)

NANONIS_SPM = Citation(
    key="nanonis_spm_2024",
    authors="Garcia, A.",
    title="nanonis-spm: Python TCP Interface for Nanonis Controllers",
    year=2024,
    citation_type=CitationType.SOFTWARE,
    url="https://pypi.org/project/nanonis-spm/",
    note="TCP communication layer",
)

NANONIS_HARDWARE = Citation(
    key="Nanonis_V5e",
    authors="Specs Zurich (Nanonis)",
    title="Nanonis Mimea V5e SPM Controller",
    year=2024,
    citation_type=CitationType.SOFTWARE,
    url="https://www.specs-zurich.com/",
    note="SPM controller hardware",
)

# ─── Autonomous STM Frameworks ─────────────────────────────────────────────

DEEPSPM = Citation(
    key="DeepSPM_2020",
    authors="Krull, P. and Hirsch, A. and Rother, C. and Schiffrin, A. and Krull, C.",
    title="Artificial-intelligence-driven scanning probe microscopy",
    year=2020,
    journal="Communications Physics",
    volume="3",
    pages="54",
    doi="10.1038/s42005-020-0317-3",
    url="https://github.com/abred/DeepSPM",
    note="CNN tip quality assessment + DQN tip conditioning",
)

SCANBOT = Citation(
    key="Scanbot_2024",
    authors="Ceddia, J. and Hellerstedt, J. and Lowe, B. and Schiffrin, A.",
    title="Scanbot: Autonomous Scanning Probe Microscopy Control",
    year=2024,
    journal="Journal of Open Source Software",
    volume="9",
    pages="6028",
    doi="10.21105/joss.06028",
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/New-Horizons-SPM/scanbot",
    note="Autonomous survey, rule-based tip conditioning",
)

NANONIS_AUTOSTM = Citation(
    key="Zhu_JACS_2024",
    authors="Zhu, Z. and Yuan, S. and Yang, Q. and Jiang, H. and Zheng, F. and Lu, J. and Sun, Q.",
    title="Autonomous Scanning Tunneling Microscopy Imaging via Deep Learning",
    year=2024,
    journal="Journal of the American Chemical Society",
    volume="146",
    pages="29199--29206",
    doi="10.1021/jacs.4c11674",
    url="https://github.com/gggg0034/Nanonis_AutoSTM_via_DL",
    note="CNN + U-Net + DQN for 48h autonomous imaging",
)

# ─── Tip Assessment & Conditioning ─────────────────────────────────────────

RASHIDI_TIP_CNN = Citation(
    key="Rashidi_2018",
    authors="Rashidi, M. and Wolkow, R. A.",
    title="Autonomous Scanning Probe Microscopy in Situ Tip Conditioning through Machine Learning",
    year=2018,
    journal="ACS Nano",
    volume="12",
    pages="5185--5189",
    doi="10.1021/acsnano.8b02208",
    note="CNN binary classification for tip quality",
)

# ─── Bayesian Optimization ─────────────────────────────────────────────────

BO_AUTOSTM = Citation(
    key="Narasimha_2024",
    authors="Narasimha, G. and Hus, S. and Biswas, A. and Vasudevan, R. and Ziatdinov, M.",
    title="Autonomous convergence of STM control parameters using Bayesian optimization",
    year=2024,
    journal="APL Machine Learning",
    volume="2",
    pages="016121",
    doi="10.1063/5.0185362",
    url="https://github.com/gnganesh99/BO-for-AutoSTM",
    note="FFT quality metric + GP/BO for bias/setpoint optimization",
)

GPCAM = Citation(
    key="gpCAM_2022",
    authors="Noack, M. M. and others",
    title="gpCAM: Gaussian Process-Based Autonomous Data Acquisition",
    year=2022,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/lbl-camera/gpCAM",
    note="General-purpose Bayesian optimization engine",
)

# ─── GP-Adaptive STS ──────────────────────────────────────────────────────

GPSTS = Citation(
    key="Thomas_2022",
    authors="Thomas, J. and others",
    title="Gaussian Process-Guided Sparse Spectroscopy for Scanning Tunneling Microscopy",
    year=2022,
    journal="npj Computational Materials",
    volume="8",
    pages="99",
    doi="10.1038/s41524-022-00777-9",
    url="https://github.com/jthomas03/gpSTS",
    note="GP regression adaptive STS sampling",
)

# ─── Atom Manipulation ─────────────────────────────────────────────────────

ATOM_MANIP_RL = Citation(
    key="Chen_2022",
    authors="Chen, I.-J. and Aapro, M. and Kipnis, A. and Ilin, A. and Liljeroth, P. and Foster, A. S.",
    title="Precise Atom Manipulation through Deep Reinforcement Learning",
    year=2022,
    journal="Nature Communications",
    volume="13",
    pages="7499",
    doi="10.1038/s41467-022-35149-w",
    url="https://github.com/SINGROUP/Atom_manipulation_with_RL",
    note="SAC + HER for atom manipulation",
)

AUTO_OSS = Citation(
    key="Wu_AutoOSS_2025",
    authors="Wu, N. and Aapro, M. and Jestil\\\"a, J. S. and Drost, R. and others",
    title="Precise Large-Scale Chemical Transformations on Surfaces: Deep Learning Meets Scanning Probe Microscopy with Interpretability",
    year=2025,
    journal="Journal of the American Chemical Society",
    volume="147",
    pages="1240--1250",
    doi="10.1021/jacs.4c14757",
    url="https://github.com/SINGROUP/AutoOSS",
    note="SAC DRL for dehalogenation reactions",
)

# ─── Molecular Recognition ─────────────────────────────────────────────────

ASD_STM = Citation(
    key="Kurki_ASD_2024",
    authors="Kurki, L. and Oinonen, N. and Foster, A. S.",
    title="Automated Structure Discovery for Scanning Tunneling Microscopy",
    year=2024,
    journal="ACS Nano",
    volume="18",
    pages="11130--11138",
    doi="10.1021/acsnano.3c12654",
    url="https://github.com/SINGROUP/ASD-STM",
    note="STM image to molecular structure prediction",
)

CARP_TOPO = Citation(
    key="Su_NatSynth_2024",
    authors="Su, J. and Li, J. and Guo, Z. and others",
    title="Intelligent synthesis of magnetic nanographenes via chemist-intuited atomic robotic probe",
    year=2024,
    journal="Nature Synthesis",
    volume="3",
    pages="466--476",
    doi="10.1038/s44160-024-00488-7",
    url="https://github.com/jiali1025/Intelligent-topological-engineering-of-quantum-p-magnets",
    note="Detectron2-based topology identification for on-surface synthesis",
)

# ─── ML Image Analysis ─────────────────────────────────────────────────────

ATOMAI = Citation(
    key="Ziatdinov_AtomAI_2022",
    authors="Ziatdinov, M. and others",
    title="AtomAI Framework for Deep and Machine Learning in Microscopy",
    year=2022,
    journal="Nature Machine Intelligence",
    volume="4",
    pages="1101--1112",
    doi="10.1038/s42256-022-00555-8",
    url="https://github.com/pycroscopy/atomai",
    note="Atomic segmentation, VAE, DKL, active learning",
)

ML_SPM = Citation(
    key="SINGROUP_mlspm",
    authors="SINGROUP",
    title="ml-spm: Machine Learning for Scanning Probe Microscopy",
    year=2023,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/SINGROUP/ml-spm",
    note="SPM image structure discovery",
)

# ─── Tip-Induced Chemistry ────────────────────────────────────────────────

ZHU_BOND_SELECTIVE = Citation(
    key="Zhu_NatComm_2026",
    authors="Zhu, Z. and others",
    title="Bond-Selective On-Surface Reactions via Autonomous STM",
    year=2026,
    journal="Nature Communications",
    volume="17",
    pages="2348",
    doi="10.1038/s41467-026-69080-1",
    note="Multi-step bond-selective manipulation",
)

# ─── LLM Integration ──────────────────────────────────────────────────────

AILA = Citation(
    key="Mandal_AILA_2025",
    authors="Mandal, A. and others",
    title="AI Laboratory Assistant for Autonomous Scientific Experimentation",
    year=2025,
    journal="Nature Communications",
    doi="",  # TODO: confirm DOI when published
    url="https://github.com/M3RG-IITD/AILA",
    note="Dual-agent LLM orchestration with LangGraph + RAG",
)

SLM_SPM = Citation(
    key="Diao_SLM_2026",
    authors="Diao, Z. and others",
    title="Small Language Model Directed SPM Operation",
    year=2026,
    url="https://github.com/DIAOZHUO/LLM-directed-SPM",
    note="LoRA fine-tuned SLM for NL to SPM command translation",
)

CLAUDE = Citation(
    key="Anthropic_Claude_2025",
    authors="Anthropic",
    title="Claude: A Family of Frontier Language Models",
    year=2025,
    citation_type=CitationType.SOFTWARE,
    url="https://www.anthropic.com/claude",
    note="LLM backbone for mission planning and data interpretation",
)

# ─── Data Processing Libraries ─────────────────────────────────────────────

SCIKIT_IMAGE = Citation(
    key="scikit_image_2014",
    authors="van der Walt, S. and others",
    title="scikit-image: Image Processing in Python",
    year=2014,
    journal="PeerJ",
    volume="2",
    pages="e453",
    doi="10.7717/peerj.453",
    note="Phase cross-correlation for drift estimation",
)

SCIPY = Citation(
    key="SciPy_2020",
    authors="Virtanen, P. and others",
    title="SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python",
    year=2020,
    journal="Nature Methods",
    volume="17",
    pages="261--272",
    doi="10.1038/s41592-019-0686-2",
    note="Signal processing, FFT, curve fitting",
)

NANONISPY = Citation(
    key="nanonispy",
    authors="Welker, J.",
    title="nanonispy: Python Module for Reading Nanonis Files",
    year=2018,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/underchemist/nanonispy",
    note=".sxm/.3ds/.dat file parsing",
)

PYSPM = Citation(
    key="pySPM",
    authors="Scholi",
    title="pySPM: Python Tools for SPM Data Analysis",
    year=2020,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/scholi/pySPM",
    note="SXM reading, plane correction, FFT analysis",
)

SPYM = Citation(
    key="spym",
    authors="rescipy-project",
    title="spym: SPM Data Analysis with xarray",
    year=2022,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/rescipy-project/spym",
    note="Line-by-line leveling, filtering",
)

SPIEPI = Citation(
    key="SPIEPy",
    authors="SPIEPy contributors",
    title="SPIEPy: Scanning Probe Image Enchanter with Python",
    year=2021,
    citation_type=CitationType.SOFTWARE,
    note="Terrace detection, surface roughness",
)

STMPY = Citation(
    key="stmpy",
    authors="Pirie, H.",
    title="stmpy: STM Data Analysis Tools",
    year=2020,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/harrispirie/stmpy",
    note="dI/dV curve fitting, gap extraction",
)

HURWITZ_FANO = Citation(
    key="HurwitzFanoFit",
    authors="Jacob, D.",
    title="HurwitzFanoFit: Fitting Kondo/Fano Line Shapes",
    year=2022,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/david-jacob/HurwitzFanoFit",
    note="Fano and Frota-Fano model fitting for Kondo physics",
)

# ─── STM Simulation ───────────────────────────────────────────────────────

PPSTM = Citation(
    key="PPSTM",
    authors="Probe-Particle",
    title="PPSTM: Probe Particle STM Simulation",
    year=2022,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/Probe-Particle/PPSTM",
    note="Tersoff-Hamann + tilted tip STM simulation",
)

# ─── Drift Correction ─────────────────────────────────────────────────────

DRIFT_CORRECTION = Citation(
    key="driftcorrection",
    authors="Lester, M.",
    title="driftcorrection: Lattice-Based STM Drift Correction",
    year=2023,
    citation_type=CitationType.SOFTWARE,
    url="https://github.com/maximelester/driftcorrection",
    note="Lattice distortion measurement and correction",
)

UNDRIFT = Citation(
    key="unDrift_2023",
    authors="various",
    title="unDrift: STM Image Drift Correction",
    year=2023,
    journal="Beilstein Journal of Nanotechnology",
    note="Cross-correlation, periodic structure, manual feature methods",
)

# ─── Adaptive PI Gain ─────────────────────────────────────────────────────

ADAPTIVE_PI = Citation(
    key="Wei_SAC_PI_2025",
    authors="Wei, T. and others",
    title="Reinforcement Learning for Adaptive PI Control in Scanning Probe Microscopy",
    year=2025,
    journal="Small",
    note="SAC for dynamic PI gain adjustment",
)


# ═══════════════════════════════════════════════════════════════════════════
#  Mapping: skill_name / function_name  →  list of Citation
# ═══════════════════════════════════════════════════════════════════════════

CITATION_DB: dict[str, list[Citation]] = {
    # ── Infrastructure (added automatically) ──
    "_mast": [MAST],
    "_nanonis_spm": [NANONIS_SPM, NANONIS_HARDWARE],
    "_claude_llm": [CLAUDE],

    # ── Builtin Skills (P0) ──
    # Basic control skills cite only infrastructure (added automatically)
    "GetBias": [],
    "SetBias": [],
    "GetCurrent": [],
    "GetZPosition": [],
    "SetSetpoint": [],
    "ZControllerOnOff": [],
    "StartScan": [],
    "StopScan": [],
    "ConfigureScan": [],
    "SetScanSpeed": [],
    "MoveToXY": [],
    "SafeRetract": [],
    "EmergencyRetract": [],
    "ConfigureLockIn": [],
    "AcquireSTS": [],
    "ConfigureSTS": [],
    "AutoApproach": [],
    "WithdrawTip": [],

    # ── Tip Assessment (P1) ──
    "AssessTipFromImage_CNN": [RASHIDI_TIP_CNN],
    "AssessTipFromImage_VGG": [DEEPSPM],
    "AssessTipFromImage_UNet": [NANONIS_AUTOSTM],

    # ── Tip Conditioning (P1) ──
    "ConditionTip_RuleBased": [SCANBOT],
    "ConditionTip_DQN": [DEEPSPM],
    "ConditionTip_DQN_v2": [NANONIS_AUTOSTM],

    # ── Region Finding (P1) ──
    "FindGoodRegion_Algo": [DEEPSPM],
    "FindGoodRegion_UNet": [NANONIS_AUTOSTM],

    # ── Resolution Optimization (P1) ──
    "OptimizeResolution_BO": [BO_AUTOSTM],
    "OptimizeResolution_gpCAM": [GPCAM],
    "AdaptivePIGain_SAC": [ADAPTIVE_PI],

    # ── Autonomous Imaging (P1-P2) ──
    "AutonomousSurvey": [SCANBOT],
    "ContinuousImaging_86h": [DEEPSPM],
    "ContinuousImaging_48h": [NANONIS_AUTOSTM],

    # ── Spectroscopy (P2) ──
    "GP_AdaptiveSTS": [GPSTS],
    "GridSTS": [],

    # ── Atom/Defect Detection (P2) ──
    "AtomDetector_FCN": [ATOMAI],
    "DefectClassifier_DKL": [ATOMAI],

    # ── Molecular Recognition (P2) ──
    "StructurePredict_ASD": [ASD_STM],
    "TopologyIdentifier_CARP": [CARP_TOPO],
    "MLspm_StructureDiscovery": [ML_SPM],

    # ── Atom Manipulation (P2) ──
    "AtomManip_SAC": [ATOM_MANIP_RL],
    "AtomManip_PathPlan": [ATOM_MANIP_RL],

    # ── Tip-Induced Chemistry (P2) ──
    "AutoOSS_Dehalogenation": [AUTO_OSS],
    "BondSelectiveReaction": [ZHU_BOND_SELECTIVE, NANONIS_AUTOSTM],
    "NanographeneSynthesis_CARP": [CARP_TOPO],

    # ── Paper-Derived Skills (mast.skills.paper) ──
    "SubtractPlane_RANSAC": [DEEPSPM],
    "LevelLines_Median": [SPYM],
    "CorrectDrift_XCorr": [SCIKIT_IMAGE],
    "FindEmptySpot": [DEEPSPM],
    "FitFano_Kondo": [HURWITZ_FANO],
    "FitGap_BCS": [STMPY],
    "AssessTip_VGG": [DEEPSPM],
    "AssessTip_ResNet": [NANONIS_AUTOSTM],
    "SegmentRegion_UNet": [NANONIS_AUTOSTM],
    "DetectAtoms_FCN": [ATOMAI],
    "PredictStructure_ASD": [ASD_STM],
    "IdentifyTopology_CARP": [CARP_TOPO],
    "OptimizeResolution_BO": [BO_AUTOSTM],
    "AdaptiveSTS_GP": [GPSTS],
    "FindGoodRegion_Heuristic": [DEEPSPM],
    "FindGoodRegion_UNet": [NANONIS_AUTOSTM],
    "ConditionTip_DQN": [DEEPSPM],
    "AutonomousSurvey_Scanbot": [SCANBOT],
    "ContinuousImaging_Auto": [DEEPSPM, NANONIS_AUTOSTM],
    "AtomManip_SAC": [ATOM_MANIP_RL],
    "AutoOSS_Dehalogenation": [AUTO_OSS],

    # ── Data Processing Functions ──
    "read_sxm": [NANONISPY],
    "read_dat": [NANONISPY],
    "read_3ds": [NANONISPY],
    "plane_subtract": [],  # standard linear algebra
    "line_by_line_level": [SPYM],
    "fft2d": [SCIPY],
    "fft_filter": [SCIPY],
    "drift_estimate": [SCIKIT_IMAGE],
    "fft_quality_score": [BO_AUTOSTM],
    "rms_roughness": [SPIEPI],
    "noise_estimate": [SCIPY],
    "DriftCorrection_Lattice": [DRIFT_CORRECTION],

    # ── Spectral Analysis ──
    "FitDIdV": [STMPY],
    "FitKondoFano": [HURWITZ_FANO],
    "GridToMap": [],

    # ── Simulation ──
    "SimulateSTM_TH": [PPSTM],
    "VAE_LatentAnalysis": [ATOMAI],

    # ── LLM Orchestration ──
    "MissionPlanner_ToolUse": [CLAUDE],
    "AILA_DualAgent": [AILA],
    "SLM_CommandTranslation": [SLM_SPM],
}

# ── Also expose a flat dict for quick key→Citation lookup ──

ALL_CITATIONS: dict[str, Citation] = {}
for _cites in CITATION_DB.values():
    for _c in _cites:
        ALL_CITATIONS[_c.key] = _c
# Add infrastructure ones too
for _c in [MAST, NANONIS_SPM, NANONIS_HARDWARE, CLAUDE, SCIPY, SCIKIT_IMAGE]:
    ALL_CITATIONS[_c.key] = _c


def get_citations_for(name: str) -> list[Citation]:
    """Look up citations for a skill or function name. Returns empty list if unknown."""
    return CITATION_DB.get(name, [])
