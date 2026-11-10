# Public corrosion data: what exists and what fits (2026-11-10)

Goal from the prof: show corrosion (and other damage) from surface or microstructure images.
Rule: public data is fine with a citation. It is never presented as our own lab data.

## Summary

| Dataset | What it is | Size / labels | Licence | Fit for us | Status |
|---|---|---|---|---|---|
| **Jackson (2025) steel corrosion** [github](https://github.com/bennett-jackson/steel-corrosion-dataset) | Bridge inspection photos of steel members, 512x512 | 3,417 imgs: good 287, fair 1,262, poor 735, severe 946, weathering 187 | MIT | Surface corrosion severity, **bridge scale** (metres). Not pits, not microstructure. | **Used, E009** |
| BMVC 2021 corrosion set (Yin et al., WPI) [code](https://github.com/Boaz-Yin/BMVC21--Corrosion-Image-Data-Set-for-Automating-Scientific-Assessment-of-MaterialsCorrosion-Image) | Lab corrosion test panels, expert-rated | 5 rating classes (per the training scripts) | see site | Closer to lab conditions | Data is hosted at arl.wpi.edu, not in the repo; not reachable from our build machine. Try from a laptop. |
| Marine corrosion dataset (Ebrahim 2025) on Kaggle | Photos of marine structures | 1,404: 1,000 healthy, 404 corroded | Apache 2.0 | Binary corroded vs healthy, structure scale | Not downloaded (Kaggle) |
| Zenodo 14653184: pitting-susceptible zones in 316L, optical monitoring | Microscope video frames of 316L polarized at +1.6 V in 5-50 mM NaCl | 6 zips, 29.4 GB, **no labels** (used for clustering) | CC BY 4.0 | Real pitting at micro scale, but unlabelled and huge | Candidate for later (unsupervised pit detection) |
| Zenodo 13269564: AA1050 pitting in ethanol fuels | **Table, not images** (115 samples, 5 conditions -> pitted yes/no) | 115 rows | CC BY 4.0 | Fits "predict corrosion from conditions", not images | Possible small tabular demo |
| Roboflow corrosion sets | Mixed photos with boxes | varies | varies | Structure scale, licences mixed | Not used |

Sources found via: [benchmarking paper using Jackson + Ebrahim](https://www.researchsquare.com/article/rs-7016003/v1), Zenodo search.

## Gap (worth saying to the prof)

We found **no public, labelled set of corrosion micrographs** (pitting vs intergranular vs uniform, or corroded vs non-corroded pits).
That is exactly the part that needs our own samples, which also makes it a real contribution.

## Proposed lab plan (needs microscope access, a few days)

1. Coupons: mild steel and 304 stainless. Grind, polish, etch, photograph "before".
2. Pitting: 3.5 % NaCl immersion, a few days.
3. Intergranular: sensitize 304 at about 650 °C, then oxalic acid etch test (ASTM A262 Practice A).
4. Uniform: dilute acid.
5. Mechanical (non-corroded) pits: punch / indent on clean coupons.
6. Photograph all with the same microscope and settings; a few hundred images in 4-5 classes.
7. Reuse the Module 1 pipeline (texture SVM, CNN, Grad-CAM) and the line-shape features (E008) for cracks.
