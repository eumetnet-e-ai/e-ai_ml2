Kapitel 5 — Neural Network Architectures
Lecture 5 — Slide Plan

1. Überblick: Warum Netzarchitekturen wichtig sind
   - Architektur = inductive bias
   - Struktur bestimmt Lernfähigkeit
   - Überblick: FFNN, GNN, CNN, LSTM

2. Feed Forward Networks: Grundidee
   - Schichtenweise Abbildung
   - Vollständig verbundene Layer
   - Regression und Klassifikation

3. FFNN Architektur
   - Input, Hidden, Output
   - Gewichte und Bias
   - Rolle der Nichtlinearität

4. FFNN in PyTorch (Minimalbeispiel)
   - nn.Linear, ReLU
   - Forward Pass
   - Trainierbare Parameter

5. Was lernt ein FFNN?
   - Funktionale Approximation
   - Parameter als Freiheitsgrade
   - Tiefe vs. Ausdrucksstärke

6. FFNN als Funktionsapproximator
   - Beispiel einer nichtlinearen Funktion
   - Wahrheit vs. NN-Approximation
   - Intuitive Interpretation

7. Computational Graph & Backpropagation
   - Vorwärts- und Rückwärtsfluss
   - Gradientenentstehung
   - Warum Training funktioniert

8. Motivation für Graph Neural Networks
   - Daten mit Relationen
   - Netze, Gitter, Physik
   - Grenzen klassischer FFNNs

9. Grundbegriffe von Graphen
   - Knoten (Nodes)
   - Kanten (Edges)
   - Features und Labels

10. Adjazenz und edge_index
    - edge_index Struktur
    - Unterschied zur Adjazenzmatrix
    - Effizienz und Skalierung

11. GNN Architekturprinzip
    - Graph Convolution
    - Nachbarschaftsaggregation
    - Nichtlinearität

12. GNN Anwendungsbeispiel
    - Advektion auf periodischem Gitter
    - Physikalische Interpretation
    - Lernziel

13. GNN Ergebnisse
    - Trainings- und Test-Loss
    - Beispielvorhersagen
    - Modellgrenzen

14. Warum Convolutional Neural Networks?
    - Lokale Muster
    - Translation-Invarianz
    - Mehrdimensionale Daten

15. CNNs für Funktionen und Zeitreihen
    - Filter statt globaler Gewichte
    - Feature Maps
    - Mehrere Skalen

16. Datengenerierung für CNNs
    - Unterschiedliche Funktionstypen
    - Rauschen
    - Klassifikationslabels

17. CNN Architektur
    - Conv1D Layer
    - Aktivierungsfunktionen
    - Klassifikationskopf

18. Training und Loss
    - Cross-Entropy Loss
    - Lernverhalten
    - Stabilität

19. Klassifikationsergebnisse
    - Korrekte vs. falsche Vorhersagen
    - Typische Fehler
    - Interpretation

20. Warum Rekurrenz?
    - Zeitabhängigkeit
    - Gedächtnis
    - Sensordaten und Prozesse

21. LSTM Zelle: Grundidee
    - Input Gate
    - Forget Gate
    - Output Gate

22. LSTM als Autoencoder
    - Rekonstruktion von Sequenzen
    - Lernen von Normalverhalten
    - Anomalien als Abweichung

23. Sensordaten und Training
    - Synthetische Daten
    - MSE Loss
    - Trainingskurve

24. Anomalieerkennung
    - Rekonstruktionsfehler
    - Schwellenwert
    - Klassifikation

25. Beispiele und Visualisierung
    - Normale vs. anomale Sequenzen
    - Interpretation
    - Grenzen des Ansatzes

26. Take-Home Messages
    - Architektur bestimmt Lernfähigkeit
    - Kein universell bestes Netz
    - Daten, Struktur und Optimierung zählen
    - Domänenwissen bleibt zentral
