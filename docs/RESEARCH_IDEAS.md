# Metallurgy × ML — idea bank (web research, 2026-10-30)

Goal: find tools that **save real time in labs / plants** (replace manual work), can be **built without lab access now**, and **demo well** to professors.
Scores 1–5 (5 = best). "Data now" = can we develop it today without our own lab.

## Score table

| # | Idea | Manual work replaced | Daily usefulness | Data now | Demo | Effort (5=easy) | Notes / evidence |
|---|---|---|---|---|---|---|---|
| 1 | **Vickers / micro-hardness indent reader** | Measuring indent diagonals by eye, computing HV | 5 | 4 | 5 | 4 | Public expert-labelled dataset (Jalilian & Uhl, RRPR 2024); DL error 1.6–1.9 % vs manual (Bertolini et al. 2026); code: Li & Yin 2021 |
| 2 | **Grain size (ASTM E112)** | Intercept / planimetric counting | 5 | 4 | 5 | 3 | Zero-shot SAM works on grains (MatSAM 2024; Liège thesis 2026); open repo ASTM-Grain-Size-Estimator (2026); MicroNet pretrained encoders (NASA, 72 % less error with 1 image) |
| 3 | **Phase / porosity / inclusion fraction (E562, E1245)** | Grid point counting | 5 | 3 | 4 | 4 | Sold as expensive microscope software (Evident/Olympus, MIPAR) → real demand; classical threshold + SAM/MicroNet |
| 4 | **XRD phase identification** | Peak search-match in vendor software, manual judgement | 5 | 5 | 4 | 3 | Open source AutoXRD (CNN trained on simulated patterns from CIF files); Simonnet et al. 2024 (identification + quantification); XMatcher 2026. Every materials lab runs XRD. |
| 5 | **Tensile-test curve analyser** | Picking 0.2 % offset yield, UTS, elongation, modulus by hand from machine CSV; spotting bad tests | 5 | 4 | 4 | 5 | Mostly deterministic + ML for slip/bad-test detection; research on NN curve evaluation (Hereon) |
| 6 | **Nodularity / nodule count (ductile iron, ASTM E2567)** | Comparing to A247 charts | 4 (foundries) | 3 | 4 | 4 | E2567 itself says automatic image analysis reduces chart variation; classical Python version exists (UFRGS 2023) |
| 7 | **Dendrite arm spacing (cast Al, superalloys)** | Manual line measurement | 3 | 2 | 3 | 3 | CNN SDAS R² 0.99 (Nikolić et al.); Faster R-CNN PDAS 98.4 % (Crystals 2021) |
| 8 | **Rust / corrosion grading from phone photo (ISO 8501-1, ASTM D610)** | Visual comparison to standard photos | 4 (field) | 3 | 5 (phone!) | 4 | CNN grading 94–96 % (2024–2026 papers); works with phone pictures → good live demo |
| 9 | **Weld radiograph defect detection** | Radiographer reading films | 4 (industry) | 3 | 4 | 3 | GDXray (small, 88 weld images); a 24,407-image weld radiograph set mentioned (Heliyon 2024) |
| 10 | **Casting X-ray porosity/shrinkage** | Radiograph reading | 4 (industry) | 3 | 4 | 3 | GDXray castings, Al-Cast dataset (ASTM classes) |
| 11 | **Fractography mode classifier (ductile / brittle / fatigue)** | Expert SEM interpretation | 3 | 1 | 4 | 3 | 98.7 % (Mater. Trans. 2025), but datasets are in-house — needs lab |
| 12 | **Surface defects (NEU)** | Strip inspection | 3 | 5 | 3 | done | Module 1, finished |
| 13 | **Metallurgy assistant (LLM + RAG over standards, SOPs, textbooks)** | Looking up etchants, standards, conversions, procedures | 4 | 4 | 4 | 3 | Claude/GPT-4o ≈ 84 % on GATE metallurgy questions vs ≤ 56 % open models (Digital Discovery 2025); hallucination → needs retrieval + citations |
| 14 | Property / Jominy predictor + inverse design | (no manual task — research tool) | 2 | 4 | 3 | 4 | NIMS fatigue, Jominy NN/GPR papers |
| 15 | Ultrasonic A-scan classification | UT operator interpretation | 3 | 1 | 2 | 2 | Mostly simulated / in-house data |
| 16 | Weld process monitoring (camera + microphone) | Post-weld inspection | 3 | 1 | 3 | 2 | Needs welding hardware (arXiv 2409.02290: 4,000 samples) |
| 17 | Plant: continuous casting inclusion prediction, LIBS scrap sorting, blast furnace | Process engineers | 3 | 1 | 2 | 2 | Needs plant data; out of scope for MTP |

## Pattern

The highest-value, buildable-now ideas are **routine lab measurements defined by ASTM standards** (1–6). They:
- replace boring, operator-dependent manual work every lab does daily;
- have a **measurable success metric**: agreement with manual measurement (%), time saved (×);
- can be developed on **public images/data now** and **validated on own samples later** (20–30 samples, manual vs app);
- don't need much training data (pretrained segmentation like SAM / MicroNet + classical image analysis + small fine-tuning).

## Recommended direction: "MetalAI Lab Assistant"

| Module | Content | Status |
|---|---|---|
| M1 | Surface defect inspector (NEU) | done — benchmark chapter |
| M2 | **Micrograph analyser**: grain size (E112), phase/porosity/inclusion fraction (E562/E1245), optional nodularity (E2567). Scale-bar calibration (µm/px), works on microscope-camera or phone-through-eyepiece images | proposed |
| M3 | **Hardness indent reader** (Vickers/Knoop → HV) | proposed |
| M4 | **Test-data tools**: tensile curve analyser (CSV → YS/UTS/El/E + bad-test flag), XRD phase ID helper (AutoXRD) | proposed (pick one or both) |
| M5 | **One-click lab report** (PDF: images, measurements, standard cited) + optional **metallurgy Q&A assistant with citations** | proposed |
| Demo extra | Rust grading from phone photo (ISO 8501) | optional |

Validation chapter (when lab access returns): same samples measured manually by 2–3 people vs the app → agreement, repeatability, time per sample.

## Sources
- Vickers: Jalilian & Uhl RRPR 2024 https://rrpr2024.sciencesconf.org/data/RRPR_2024_paper_1.pdf · Bertolini et al. 2026 https://aimnet.it/la_metallurgia_italiana/2026/gennaio/Aim_gennaio2026_1.pdf · Li & Yin code https://github.com/SimonLeeCHN/IndentMes_code_ref
- Grain size: arXiv 2604.18957 https://arxiv.org/abs/2604.18957 · repo https://github.com/mueez-overflow/ASTM-Grain-Size-Estimator · MatSAM https://arxiv.org/abs/2401.05638v2 · SAM grains (Seagate) https://arxiv.org/abs/2408.12732 · Liège thesis https://matheo.uliege.be/handle/2268.2/25195
- MicroNet (NASA): https://www.nature.com/articles/s41524-022-00878-5 · CS-UNet https://ar5iv.labs.arxiv.org/html/2308.13917
- Phase/inclusions vendors: https://evidentscientific.com/en/applications/nonmetallic-inclusion-analysis-in-steel · ASTM E1245 https://standards.iteh.ai/catalog/standards/astm/2b6c506e-e470-463a-8cad-c857b8bfe7b4/astm-e1245-03-2023
- Nodularity: ASTM E2567 https://store.astm.org/e2567-14.html · UFRGS https://lume.ufrgs.br/handle/10183/267751
- XRD: AutoXRD https://pypi.org/project/autoXRD/ · XMatcher https://arxiv.org/abs/2607.17162 · XERUS https://arxiv.org/abs/2112.04773v1
- Tensile: Hereon https://publications.hereon.de/id/50781/index.php · small punch DL https://web3.arxiv.org/pdf/2506.17680
- Corrosion: https://arxiv.org/pdf/2008.05204 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11175235/ · https://iris.univpm.it/retrieve/56db5643-9d6e-470d-916e-7445ebcda3e9/Mandolini_Automatic-assessment-rust-level_2026.pdf
- Weld/casting X-ray: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11079250/ · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10943115/ · https://arxiv.org/pdf/1808.02518.pdf · https://arxiv.org/pdf/2209.02279
- Fractography: https://www.jstage.jst.go.jp/article/matertrans/66/10/66_MT-Z2025004/_article · https://arxiv.org/pdf/2007.02267
- Dendrites: https://doaj.org/article/4b7b525b17e64747b71ffd7406abda13 · https://www.bib.irb.hr/1127638
- LLMs: https://pubs.rsc.org/en/Content/ArticleLanding/2025/DD/D4DD00319E · https://arxiv.org/pdf/2310.07197 · https://arxiv.org/pdf/2504.14849
- Jominy: https://www.iris.sssup.it/retrieve/859042bb-4a7c-42e9-a79c-4fb3a7059dbe/IWANN_2023_paper_1136.pdf
- UT: https://arxiv.org/pdf/2112.06650 · Welding monitoring: https://arxiv.org/html/2409.02290v1
- Plant: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10422374/ · LIBS: https://ui.adsabs.harvard.edu/abs/2017AcSpB.134...52C/abstract
- Phone microscopy: https://www.machinedesign.com/news/holder-lets-smartphones-take-pictures-microscopic-images

---

# Part B — "Wow" demos to impress professors (research 2026-10-30)

Criteria: visual + interactive, clearly metallurgical, runnable on the RTX 4060 laptop, data/code available.

| # | Demo | What the professor sees | Wow | Metallurgy | Feasible on laptop | Data / code |
|---|---|---|---|---|---|---|
| W1 | **Virtual heat-treatment lab** (CCT/TTT predictor + cooling-curve simulator) | Pick a steel composition → AI draws its CCT diagram → drag a cooling curve → app shows which phases form (ferrite/pearlite/bainite/martensite %) and predicted hardness | 5 | 5 | yes (CPU) | ML on CCT/TTT data beats JMatPro on test samples (Huang 2023); NN CCT tools exist (Vermeulen 1997, Polish NN program); physics baseline = empirical Kirkaldy/Li equations if data is thin |
| W2 | **AI microstructure generator** | Slide annealing temperature/time → a realistic micrograph appears | 5 | 4 | yes (8 GB, LoRA) | Stable Diffusion + LoRA on UHCSDB (598 imgs with processing metadata); NIST UHCSDB |
| W3 | **2D → 3D microstructure** (SliceGAN) | Upload one micrograph → rotate a generated 3D block of that microstructure in the browser | 5 | 4 | yes (single-image training) | SliceGAN, Nature Mach. Intell. 2021, code public; 10^8-voxel volume in seconds |
| W4 | **AI grain-growth / solidification simulator** (phase-field surrogate) | Click to place nuclei → grains grow / dendrites solidify live, ~100× faster than the physics solver; compare AI vs physics side by side | 5 | 5 | yes (train data from own small phase-field code) | U-Net grain-growth surrogate (arXiv 2205.02121), dendrite surrogate ~100× (LLNL 2025), CNN from one small sim (arXiv 2511.10171) |
| W5 | **Atom-scale virtual lab** (universal ML interatomic potentials) | Build an Fe/Al/Cu crystal → lattice constant, elastic constants, vacancy energy, melting MD in minutes instead of days of DFT | 4 | 4 | yes (MACE-MP-0 small / CHGNet) | MACE-MP-0, CHGNet; caveat: alloy mixing energies & some mechanical props inaccurate → show benchmark vs known values |
| W6 | **Micrograph → hardness** | Upload a micrograph → predicted hardness / strength + Grad-CAM on the phase responsible | 4 | 5 | yes | CNN micro-hardness from microstructure (IJAMT 2022), tempered martensite image regression (Results in Materials 2024), tensile strength from LOM (ICCS 2023) — needs paired image+property data |
| W7 | LLM alloy-design agent | Chat → proposes HEA compositions, computes δ/VEC/ΔH, predicts phase | 4 | 4 | yes (API) | ReAct HEA agent arXiv 2603.11068 |
| W8 | Crystal GNN / generative crystals (CGCNN, ALIGNN, MatterGen) | Predict moduli of crystals / generate new crystals | 3 | 2 (chemistry-heavy) | partly | Matbench, Materials Project, MatterGen (Nature 2025) |
| W9 | Autonomous-lab style active learning | AI chooses next experiment (Bayesian optimisation) on a simulated process | 3 | 3 | yes | A-Lab (Nature 2023; results later disputed — good discussion point) |

## Recommended "Virtual Metallurgy Lab" package (pick 2–3)
1. **W1 Heat-treatment lab** — the most metallurgical and interactive; every professor teaches CCT diagrams.
2. **W4 AI grain-growth / solidification simulator** — live, visual, physics vs AI side by side.
3. **W3 2D→3D microstructure** or **W2 microstructure generator** — the visual "wow".
Module 1 (surface defects) stays as the computer-vision chapter.

## Sources (Part B)
- SliceGAN: https://arxiv.org/abs/2102.07708v1 · https://spiral.imperial.ac.uk/entities/publication/8f1da115-af65-4b4b-9ac4-a98a1c80db8d
- UMLIPs: https://arxiv.org/pdf/2503.09814 · https://arxiv.org/html/2506.21935v2 · https://arxiv.org/pdf/2504.06993
- CCT/TTT ML: https://link.springer.com/content/pdf/10.1007/s42243-023-00932-6.pdf · https://link.springer.com/article/10.1007/s11837-020-04057-z · https://mdr.nims.go.jp/datasets/c5548153-dfa1-475b-ab9a-aee2b691d3b0.md · https://repozytorium.biblos.pk.edu.pl/redo/resources/33927/file/suwFiles/JagielloA_ComputerSoftware.pdf
- Phase-field surrogates: https://arxiv.org/pdf/2511.03884 · https://arxiv.org/pdf/2205.02121 · https://arxiv.org/abs/2511.10171v1 · https://www.osti.gov/pages/biblio/2461606-deep-operator-network-surrogate-phase-field-modeling-metal-grain-growth-during-solidification
- Microstructure diffusion: https://www.themoonlight.io/en/review/process-aware-and-high-fidelity-microstructure-generation-using-stable-diffusion · https://materialsdata.nist.gov/handle/11256/964
- Micrograph→property: https://link.springer.com/article/10.1007/s00170-022-10355-4 · https://doaj.org/article/045884ca44124c95bb26cd5b2ddabcb1 · https://www.iccs-meeting.org/archive/iccs2023/papers/140770504.pdf
- MatterGen: https://arxiv.org/abs/2312.03687v2 · CGCNN https://arxiv.org/pdf/1710.10324v2
- A-Lab: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10700133/ · critique https://venturebeat.com/ai/ai-meets-materials-science-the-promise-and-pitfalls-of-automated-discovery
- PINN quenching: https://search.napstic.cn/literature/periodical/083b4e9b4bc9346e9f1a6a354ac87483 · https://api.crossref.org/works/10.1007%2FS00366-025-02234-9
