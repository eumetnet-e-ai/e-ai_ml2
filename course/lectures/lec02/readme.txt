Folie 1: Titel / Zielbild (Notebooks, APIs, Remote-Setup, Native Code Integration)
- Fokus: “Produktiv arbeiten” statt UI-Details

Folie 2: Warum Notebooks im ML/Science-Workflow (DWD/Research-Kontext)
- Exploration + Dokumentation + Visualisierung
- Schnell iterieren, aber sauber reproduzierbar halten

Folie 3: Jupyter Architektur in 1 Bild: Browser ↔ Server ↔ Kernel
- Stateful execution als Stärke/Risiko

Folie 4: Reproduzierbarkeit in Notebooks: 4 Regeln
- Kernel restart + Run All
- klare Imports/Parameter-Zellen
- Outputs speichern (Plots/Artefakte)
- Abhängigkeiten dokumentieren

Folie 5: Environments & Pakete: der eine sichere Weg
- !{sys.executable} -m pip install ...
- sys.executable als Diagnose

Folie 6: Markdown & “Narrative Computing” (Minimum)
- Überschriften, Bulletlists, LaTeX für Formeln
- Ziel: Notebook als lesbares Experimentprotokoll

Folie 7: Magic + Shell: die 5 Befehle, die man wirklich nutzt
- %timeit, %%writefile, %%bash, !pip, !ls

Folie 8: Visualisierung als Qualitätskontrolle (Matplotlib/Seaborn/Plotly)
- Statisch vs interaktiv: wann was?

Folie 9: Demo 1 (3 min): Lorenz63 Notebook – laufen lassen, Parameter ändern, Plot speichern
- Zeigt Jupyter-Workflow ohne “Klickkunde”-Folien

Folie 10: Von Notebook zu Engineering: Warum APIs als Strukturprinzip
- Trennung von Zuständigkeiten, wiederverwendbare Bausteine

Folie 11: API-Typen (aus deinem Skript): lokal / Library / Web
- Lokale Module als “eigene API”
- Library APIs (numpy/pandas)
- Web APIs (HTTP)

Folie 12: REST Essentials: Ressourcen, Methoden, Statuscodes
- GET/POST/PUT/DELETE
- 200/201/400/404 (Minimum)

Folie 13: Flask Server im Beispiel: Endpoints (Items CRUD)
- /items, /items/<id>
- Validierung: abort(400/404)

Folie 14: Client mit requests: Muster & Debugging
- json= senden, status_code prüfen
- Fehler schnell lesen: 400 vs 404

Folie 15: Demo 2 (3 min): REST einmal durchspielen (GET/POST/DELETE)
- Server starten, 2–3 Calls zeigen (oder Client-Script)

Folie 16: Upload/Download via REST (nur 1 Folie, kein Overhead)
- Upload: files={'file': open(...,'rb')}
- Download: stream=True + chunks schreiben
- typische Stolpersteine: Pfad/working dir/allowed extensions

Folie 17: Warum Fortran/C++ in Python/Notebooks einbinden? (DWD-relevant)
- Legacy-Code (Fortran), Performance, Validierung gegen Referenz
- Python als Orchestrierung + Visualisierung + Glue

Folie 18: Optionen im Überblick (Entscheidungsmatrix)
- ctypes (shared lib, C-ABI, leichtgewichtig)
- f2py (Fortran→Python Wrapper, sehr bequem)
- pybind11 (C++→Python, modern, robust)
- (optional mention) CFFI/SWIG (nur als Stichwort)

Folie 19: Fortran-API sauber machen: iso_c_binding + bind(C)
- C-kompatible Signatur → von Python ladbar
- Datentypen: real(c_double) etc.

Folie 20: Minimalbeispiel Fortran → .so bauen (Build-Snippet)
- Fortran code (f_sin_cos)
- Compile: gfortran -shared -fPIC fortran_interface.f90 -o fortran_interface.so
- Hinweis Windows: .dll (nur erwähnen)

Folie 21: Python-Seite: Laden & Aufrufen via ctypes (Notebook-tauglich)
- ctypes.CDLL("./fortran_interface.so")
- argtypes/restype setzen (wichtig!)
- Werte testen + plotten

Folie 22: C++-Alternative: pybind11 in 90 Sekunden (wann sinnvoll?)
- Wenn C++-Klassen/Objekte, komplexere Strukturen, Exceptions, RAII
- Mini-Skizze: C++ Modul → import mymodule in Python
- (keine Build-Details, nur Konzept + Nutzen)

Folie 23: Remote Jupyter: Prinzip (Compute remote, Browser lokal)
- Sicherheit: nicht “8888 ins Netz öffnen”, lieber SSH-Tunnel

Folie 24: Port Forwarding Rezept (Copy/Paste) + Token
- Remote: jupyter notebook --no-browser --port=8888
- Local: ssh -N -L 9001:localhost:8888 user@remote-host
- Browser: http://localhost:9001 (Token aus Log; Port ersetzen)

Folie 25: Firewalls & typische Probleme (Troubleshooting-Checkliste)
- Port belegt lokal/remote
- SSH erlaubt, 8888 inbound blockiert (ist okay beim Tunnel)
- Bastion/Jump host falls nötig
- Empfehlung: --ip=127.0.0.1 auf remote, keine offenen Notebook-Ports
