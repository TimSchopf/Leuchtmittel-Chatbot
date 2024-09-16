# Leuchtmittel-Chatbot

### Coding Challenge für einen Leuchtmittel Chatbot

**Ziel:** Entwicklung einer Suchlösung auf Basis von PDF-Dokumenten unter Verwendung eines Language Model (LLM).

**Teil 1:** RAG (Retrieval Augmented Generation) mit PDF-Dokumenten:
Dir werden PDF-Dokumente mit definiertem Inhalt bereitgestellt.

_Such-Tool Entwicklung:_ Erstelle ein Stück Software, in dem du Suchanfragen von Nutzern eingeben kannst. Es reicht
dabei, wenn das im Code direkt passiert, es muss keine UI o.ä. entwickelt werden. Es ist ebenfalls kein Interface für
das Hochladen oder Auswählen der PDFs nötig, auf diese kann einfach per Dateisystem an fixer Stelle zugegriffen werden.

_Retrieval:_ Identifiziere relevante Abschnitte in den PDF-Dokumenten basierend auf den Benutzeranfragen. Es reicht
dabei, wenn die produzierten Daten nur flüchtig vorgehalten werden, es muss nicht zwingend persistiert werden.

_Antwort-Generierung:_ Generiere und präsentiere die Antworten mit Hilfe der Ergebnisse der Suche und des LLM. Es reicht
auch hierbei ein einfacher Output auf stdout o.ä..

**Teil 2:** Qualitätssicherung:

Damit du deinen Ansatz validieren kannst, hier ein paar Fragen, die wir deinem System sehr wahrscheinlich im weiteren
Interview-Verlauf stellen werden:

Wie viel wiegt XBO 4000 W/HS XL OFR?

Welche Leuchte hat SCIP Nummer dd2ddf15-037b-4473-8156-97498e721fb3?

Welche Leuchte hat die Erzeugnissnummer 4008321299963?

Gebe mir alle Leuchtmittel mit mindestens 1500W und einer Lebensdauer von mehr als 3000 Stunden?

Versuche sicherzustellen, dass solche Anfragen korrekt beantwortet werden.

**Die Quell-PDFs findest du hier:
** https://drive.google.com/drive/folders/1z2gqtLxgnFzFkGNpURnOMbgdezLQ-KoD?usp=share_link

## Projekt Ausführen

* Installiere die Requirements `pip install -r requiremens.txt`
* Füge im `.streamlit` Ordner eine `secrets.toml` Datei mit einen gültigen OpenAI API key hinzu.
  Siehe: `.streamlit/secrets-example.toml`.
* Starte die Streamlit UI `streamlit run ui.py`
