# Phase 3A STT benchmark samples

The audio files in this directory are **not committed** (downloaded artifacts,
see `.gitignore`). To reproduce the Phase 3A benchmark:

1. Download `hindi.ogg` (Hindi, no ground truth exists) and decode it to
   `samples/hindi.wav` (mono, 16 kHz) with any converter (PyAV, ffmpeg, Audacity):

   ```powershell
   Invoke-WebRequest -Uri "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/hindi.ogg" -Headers @{ "User-Agent" = "IndicVoiceRAG/0.1" } -OutFile "$env:TEMP\hindi.ogg"
   ```

   Sidecar `samples/hindi.txt` (empty first line = no ground truth, so WER is
   reported as `accuracy NOT MEASURED` - the dataset ships no transcripts):

   ```text
   language = hi
   source = Narsil/asr_dummy:hindi.ogg
   ```

2. Download the LibriSpeech dummy parquet and write each row you want as
   `samples/en_<id>.wav` with the real `text` column on the first line of its
   sidecar:

   ```powershell
   Invoke-WebRequest -Uri "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/validation/dev_clean.parquet" -Headers @{ "User-Agent" = "IndicVoiceRAG/0.1" } -OutFile "$env:TEMP\libri.parquet"
   ```

   ```text
   MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
   language = en
   source = hf-internal-testing/librispeech_asr_dummy
   ```

3. Run:

   ```powershell
   .venv\Scripts\python.exe -m indicvoicerag.cli benchmark-stt --samples-dir samples --repeat 2 --out results/stt_benchmark.json
   ```

Sidecar format: first line = ground-truth transcript (blank = none),
`language = <code>` (pinned language passed to the model),
`source = <where it came from>`.

Audio is excluded from git; this file and the report in `results/` are what
get committed.
