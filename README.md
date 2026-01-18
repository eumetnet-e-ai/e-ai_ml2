# E-AI — AI/ML Intensive Training Course (ECMWF, January 2026)

This repository contains the material for the **E-AI AI/ML Intensive Training Course**, developed in a first version in 2025 at DWD and given to 160 participants, revised and extended and delivered at **ECMWF in January 2026** for 220+ participants from **ecmwf staff**.

The course provides a structured, hands-on introduction to **modern AI/ML methods** with a strong focus on **weather and climate applications**, including operational perspectives, reproducibility, and best practices for building reliable ML workflows.

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
| 12 | 3 | 12 | MLFlow – Managing and Monitoring Training |
| 13 | 4 | 13 | MLOps – Development and Operations Integrated |
| 14 | 4 | 14 | CI/CD – Continuous Integration and Deployment |
| 15 | 4 | 15 | Anemoi – AI-Based Weather Modeling |
| 16 | 4 | 16 | The AI Transformation |
| 17 | 5 | 17 | Model Emulators, AIFS and AICON |
| 18 | 5 | 18 | AI Data Assimilation |
| 19 | 5 | 19 | AI and Physics and Data |
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
- Use the manuscript PDF for further details!
- Exercises are designed to be runnable on standard laptops. Eccodes will need LINUX or MAC. 

### Trainers / Instructors
- Slides are built via LaTeX.
- Notebooks are designed to run sequentially.
- Figures are referenced via relative paths (see `images/`).

---

## License

This material is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

---

## Authors and Contributions

The material has been generated mainly by **Roland Potthast**, with contributions and support from:

- Stefanie Hollborn
- Jan Keller
- Marek Jacob
- Florian Prill
- Tobias Göcke
- Felix Fundel
- Thomas Deppisch
- Mareike Burba
- Matthias Mages
- Sarah Heibutzki

---

## Acknowledgements

This course builds on the work and experience of the E-AI programme and the broader weather/climate AI community, including operational NWP workflows and open-source software ecosystems.

---
