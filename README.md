# E-AI — AI/ML Intensive Training Course 

## ECMWF -- University of Reading -- Deutscher Wetterdienst (DWD)
### January 2026

This repository contains the material for the **E-AI AI/ML Intensive Training Course**. The first version was developed at **DWD in 2025** and delivered to **160 participants**. Parts of the course were given at the E-AI Summer Workshop in July, 2025, with about 80 people participating. The course was also delivered to a team of **NWP scientists in Oman in November 2025**. It was revised and extended and then delivered at **ECMWF in January 2026** for **220+ participants from ECMWF staff**.

The course provides a structured, hands-on introduction to **modern AI/ML methods** with a strong focus on **weather and climate applications**, including operational perspectives, reproducibility, and best practices for building reliable ML workflows. 

The course expects participants to actively engage with the material. It is designed to build intuition through **simple hands-on examples** for all key techniques and to provide **practical insight wherever possible**.

---

## Scope and Learning Goals

After completing the course, participants will be able to:

- work confidently with **Python** and **Jupyter notebooks** for AI/ML workflows
- understand core ML concepts: models, loss functions, optimization, generalization
- implement and train neural networks in **PyTorch**
- understand **LLMs** and modern workflows such as **RAG**, **tool/function calling**, and **agents**
- apply AI/ML methods to typical meteorological data formats (GRIB, NetCDF, OpenData)
- connect AI/ML workflows to operational themes: verification, monitoring, reproducibility, MLOps/CI
- gain an overview of AI weather models (e.g. **Anemoi, AIFS, AICON**) and AI-based data assimilation
- understand the role of **physics and domain knowledge** in AI models (e.g. **physics-informed learning**, constraints, hybrid modelling)
- understand how **forecasting from observations** works, including **data-driven nowcasting/short-range prediction** and learned dynamics


---

## Course Structure (5 Days)

The training is organized as **5 days**, with **20 sessions** (Chapters 1–20). Each session is complemented by a lab session, where codes are run and discussed in small groups.

| Session | Day | Chapter | Title |
|---:|---:|---:|---|
| 01 | 1 | 1 | Python Basics |
| 02 | 1 | 2 | Jupyter Notebooks, APIs and Servers |
| 03 | 1 | 3 | Eccodes for GRIB, OpenData, NetCDF, Visualization |
| 04 | 1 | 4 | Basics of Artificial Intelligence and Machine Learning (AI/ML) |
| 05 | 2 | 5 | Neural Network Architectures |
| 06 | 2 | 6 | Large Language Models |
| 07 | 2 | 7 | LLM with Retrieval-Augmented Generation (RAG) |
| 08 | 2 | 8 | Multimodal LLMs |
| 09 | 3 | 9 | Diffusion and Flexible Graph Networks |
| 10 | 3 | 10 | Agents and Coding with LLM |
| 11 | 3 | 11 | DAWID, LLMs and Feature Detection |
| 12 | 3 | 12 | MLflow – Managing and Monitoring Training |
| 13 | 4 | 13 | MLOps – Development and Operations Integrated |
| 14 | 4 | 14 | CI/CD – Continuous Integration and Deployment |
| 15 | 4 | 15 | Anemoi – AI-Based Weather Modeling |
| 16 | 4 | 16 | The AI Transformation |
| 17 | 5 | 17 | Model Emulators, AIFS and AICON |
| 18 | 5 | 18 | AI Data Assimilation |
| 19 | 5 | 19 | AI, Physics, and Data |
| 20 | 5 | 20 | Learning from Observations Only |

Appendix (optional): **History of Large Language Models**.

---

## Repository Contents

Typical contents include:

- **Slides** (LaTeX sources and PDFs) for lectures (lec01–lec20)
- **Jupyter notebooks** with demos, exercises, and reference workflows
- **Figures and graphics** used in the course
- supporting **scripts** for data access and processing
- `requirements.txt` / environment definitions

Large datasets are generally **not stored in Git**. See `data/` notes if present.

---

## How to Use the Material

### Participants
- Follow the course schedule (Day 1 → Day 5).
- Use the lecture PDFs as orientation and run the associated notebooks.
- Use the manuscript PDF for further details.
- Exercises are designed to run on standard laptops. **ECCODES requires Linux or macOS.**

### Trainers / Instructors
- Slides are built via LaTeX.
- Notebooks are designed to run sequentially.
- Figures are referenced via relative paths (see `images/`).

---

## License

This work is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

You are free to share and adapt the material for any purpose, even commercially, **as long as you give appropriate credit**.

© Roland Potthast and contributors, 2025–2026.

CC BY 4.0: https://creativecommons.org/licenses/by/4.0/


You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

## Attribution

If you reuse this material, please cite:

Roland Potthast et al., *EUMETNET E-AI ML2 Course Material*, CC BY 4.0.

---

## Authors and Contributions

The material has been generated mainly by **Roland Potthast**, with contributions and support from:

- Stefanie Hollborn (FRAIM, Transformer, LLM)
- Jan Keller (AI-VAR, AICON)
- Marek Jacob (Anemoi Product Application Example, AICON, Python)
- Florian Prill (AICON Walkthrough)
- Tobias Göcke (AICON)
- Felix Fundel (Front Detection, Verification)
- Thomas Deppisch (FRAIM, AI-VAR)
- Mareike Burba (Python)
- Matthias Mages (FRAIM)
- Sarah Heibutzki (AI-VAR)
- Nikolas Porz (DAWID)

---

## Acknowledgements

This course builds on the work and experience of the E-AI programme and the broader weather/climate AI community, including operational NWP workflows and open-source software ecosystems.

---
