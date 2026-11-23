# ATC Copilot

A real-time system that listens to air traffic control radio, turns each transmission into a structured clearance, and checks it against what the aircraft actually does.

## The one-line pitch

"I built an open, streaming pipeline that transcribes ATC radio, parses clearances, and checks them against what the aircraft broadcasts over ADS-B: its track and the altitude its crew set in the autopilot. It flags readback errors and non-conformance within seconds."

## Why this project

- **It fills gaps in the portfolio.** Existing work covers simulation and imitation learning (drivesim), agents and retrieval (Trace, the support agent), and multimodal pre-training. This adds speech, noisy real-world data, and a streaming system with a hard latency budget.
- **The problem is real.** Readback and hearback errors (a pilot repeats an instruction wrongly and the controller misses it) are a known contributor to runway incursions and altitude busts. FAA studies find errors in under 1% of messages [6], but with millions of clearances a day that adds up.
- **Ground truth is free.** Many aircraft broadcast their position, their altitude, and the altitude, heading, and altimeter setting the crew have selected. A parsed "climb to flight level 240" can be checked against what the crew dialled in and what the plane did.
- **The demo is visual.** A live map of one airport with the transcript scrolling next to the aircraft it refers to.

## What is new here, and what is not

Every individual piece has prior work, including readback error detection itself. The contribution is the combination, the open models, and the use of surveillance data as the check.

| Piece | Prior work | This project's contribution |
|---|---|---|
| Whisper fine-tuned on ATC | Whisper-ATC (van Doorn et al., ICRAT 2024): 13.5% WER on ATCO2, open models on Hugging Face [1] | Use it as the baseline; measure how well it transfers to the demo airport |
| Callsign biasing from surveillance data | Kocour et al., Interspeech 2021: OpenSky callsign lists boosted callsign accuracy by 27 points absolute, to 82.9% [2] | Reproduce it with Whisper hotwords or N-best rescoring instead of a WFST decoder |
| Speaker role detection | BERTraffic (Zuluaga-Gomez et al., SLT 2022) [3] | Compare rules, text model, and audio features on the demo airport |
| Readback error detection from speech | HAAWAII project (Helmke et al., DLR, 2021 to 2022): 81% detection on real Icelandic recordings and about 11% false alarms in controller trials, with a closed, operational setup [4][5] | Open models on public-domain inputs, and a **second, independent check against ADS-B selected altitude and track**, which HAAWAII did not use |

HAAWAII's main lesson applies directly: comparing controller and pilot transcripts word by word did not work. Comparing them at the level of parsed instructions ("concepts") did. So the checker compares parsed instructions, never raw text.

Read [1] through [5] before Week 1 so the baselines and terminology match the literature, and quote HAAWAII's numbers in the write-up as the reference point. Note that their data, airspace, and error sets differ from yours.

## How it works

```
Airband audio (own SDR receiver; LiveATC for listening during development only)
        |
        v
1. Segmenter      voice activity detection splits the stream into transmissions
        |
        v
2. Speech-to-text Whisper fine-tuned for ATC, biased with callsigns in range
        |                                        ^
        v                                        |
3. Parser         transcript -> {speaker, callsign, instructions}
        |                                        |
        v                                        |
4. Linker         match callsign to a live ADS-B track  <--- ADS-B (own receiver, or OpenSky)
        |
        v
5. Checker        a) readback vs instruction
        |         b) instruction vs selected altitude / heading / QNH
        |         c) instruction vs actual track
        v
6. Live map UI    aircraft, transcripts, flagged events
```

### 1. Segmenter

ATC radio is push-to-talk, so transmissions are short bursts separated by silence or squelch. Silero VAD (or an energy threshold tuned to the squelch) cuts the stream into clips of roughly 2 to 10 seconds. Good segmentation matters more than it looks: Whisper hallucinates text on noise and silence, and a clip that merges two speakers breaks the parser. Drop clips under about half a second, and discard transcripts whose average token log-probability or no-speech probability falls outside a tuned threshold.

### 2. Speech-to-text

Stock Whisper does poorly on ATC audio: narrowband radio, heavy noise, fast speech, non-native accents, and a vocabulary of callsigns, waypoints, and digits spoken one by one ("niner", "flight level two four zero").

- **Baselines:** stock Whisper, and Whisper-ATC (`jlvdoorn/whisper-large-v3-atco2-asr` and its ATCOSIM variant) [1]. Beating Whisper-ATC is the real test. Its model card doesn't document the training data, so assume it may have seen parts of the ATCO2 test material and lean on your own airport's test set.
- **Fine-tune** `small` or `medium` first, `large-v3` or `large-v3-turbo` if compute allows, on public corpora plus a small amount of labelled audio from your airport. Keep the private test set out of training.
- **Serve** with `faster-whisper` for low latency.
- **Bias decoding with callsigns in range.** The Whisper prompt is limited to about 224 tokens, which holds a busy airport's callsign list but not much else. Compare three methods:
  1. prompting with the spoken forms ("Delta twelve thirty-four"),
  2. `faster-whisper` hotwords,
  3. rescoring the N-best list against the callsign list.

  Report callsign accuracy for each, with and without biasing, against Kocour et al. [2].
- **Fix the text normalisation first.** Decide whether references say "flight level two four zero" or "FL240", and normalise hypotheses and references the same way. WER numbers are not comparable across papers or corpora otherwise. Report the normalisation rules in the write-up.
- **Get calibrated confidence.** The checker relies on confidence, so calibrate per-utterance and per-number-span scores from token log-probabilities on the labelled set (reliability diagram, expected calibration error).
- **Watch the domain shift.** The public corpora are European, ICAO phraseology. A US airport uses FAA phraseology ("descend and maintain", altitudes in feet below 18,000 ft and flight levels above, different callsign habits). Measure on your own airport early, not just on the public test sets.

### 3. Parser

Convert the transcript into a structured record:

```json
{
  "speaker": "controller",
  "callsign": "DAL1234",
  "instructions": [
    {"type": "climb", "altitude_ft": 24000, "reference": "FL"},
    {"type": "heading", "degrees": 270, "turn": "left"},
    {"type": "contact", "frequency_mhz": 124.35}
  ],
  "asr_confidence": 0.91,
  "value_confidence": {"altitude_ft": 0.95, "degrees": 0.62}
}
```

Compare two approaches; the comparison is part of the write-up:

- **Grammar-based.** Phraseology is standardised, so rules cover most of it, and they are fast and predictable.
- **Model-based.** A small fine-tuned model, or an LLM with constrained JSON output, for the non-standard phrasing rules miss. Report its latency as well as its F1, because it sits in the live path.

**Callsigns.** Spoken callsigns need mapping to ICAO form: "Delta twelve thirty-four" becomes `DAL1234`. Build an airline telephony table ("Speedbird" is `BAW`, "Brickyard" is `RPA`), plus rules for general aviation callsigns ("November one two three alpha bravo", "Cessna three alpha bravo"). Seed the table from the callsigns actually seen on ADS-B at your airport.

**Speaker role.** Controller and pilot share the channel. Controllers usually lead with the callsign and pilots usually end with it, which gets a rules baseline most of the way. BERTraffic [3] shows a text model does better. Audio features help too: at many airports the controller's transmitter and the aircraft are received at very different signal levels and quality. Pick the airport partly on how separable the two sound.

### 4. Linker

Match the parsed callsign to an aircraft currently in range. Use fuzzy matching against the live callsign list, weighted by how plausible each aircraft is: is it in this frequency's area, and was it spoken to recently? Keep each transmission's top few candidates rather than committing early; a later readback often settles it. Report linker accuracy separately from ASR callsign accuracy.

### 5. Checker

Three independent checks. A flag backed by two of them is much more trustworthy than one backed by one, so show which checks fired with each event.

- **Readback check.** Pair a controller instruction with the pilot reply that follows it. Compare the parsed instructions, not the words. Flag a mismatch in any value (altitude, heading, runway, frequency, squawk, altimeter setting) or a missing element. Heading and frequency changes are the most common readback errors in FAA data [7], and errors grow with the number of items in a message, so test those cases hardest. Flag only when the relevant number spans in both transcripts are high confidence.
- **Selected-value check.** Many aircraft broadcast the values set in the autopilot. `readsb` decodes them as `nav_altitude_mcp` (selected altitude), `nav_heading` (selected heading, usually magnetic), and `nav_qnh` (altimeter setting) [8]. This is the strongest signal in the project: a wrong selection shows up seconds after the clearance, before the aircraft goes anywhere. It works even when the pilot's readback is garbled. OpenSky's standard state vectors do not include these fields [9], which is the main reason for your own receiver. Report coverage, meaning the share of aircraft at your airport that broadcast each field.
- **Track conformance.** After a clearance, watch the track. Flag if the aircraft does not start complying within a time window, levels off at the wrong altitude, or turns the wrong way. Two corrections are needed:
  - ADS-B barometric altitude is pressure altitude, referenced to 1013.25 hPa / 29.92 inHg. Below the transition altitude (18,000 ft in the US), clearances use local altimeter settings. Correct with the current METAR, or you will see false busts on every pressure day. The broadcast `nav_qnh` and geometric altitude give a cross-check.
  - ADS-B gives ground track, not heading, and true rather than magnetic. Apply the local magnetic variation, and allow for wind with a tolerance band or a wind estimate from the traffic.
- **Event detection.** Go-arounds, holds, and rejected takeoffs show up in both the audio and the track, and they make good demo moments.

### 6. Live map UI

A web page with a map (MapLibre or deck.gl) centred on one airport. Aircraft move in real time; clicking one shows its transcript history and parsed clearances. Flagged events appear in a side feed with the audio clip and the evidence: what was said, what was read back, what was selected, and what the aircraft did.

**Latency budget** (target under 3 s from end of transmission to event):

| Stage | Budget |
|---|---|
| VAD end-of-speech detection | 300 ms |
| ASR (`faster-whisper` on GPU, 5 s clip) | 500 to 1,000 ms |
| Parser (rules, or small model) | under 50 ms, or under 500 ms |
| Linker and readback check | under 50 ms |
| UI push | under 200 ms |

Selected-value and conformance checks run later by nature; report their time to flag separately.

## Data

| Source | What it gives | Notes |
|---|---|---|
| ATCO2 | Real ATC radio with transcripts, several European airports | A 1-hour test set is free; the 4-hour test set and a ~5,300-hour automatically transcribed set are sold through ELDA [10] |
| UWB-ATCC | About 20 hours of Czech-airspace ATC recordings in English (roughly 10 hours of speech after removing silence) | CC BY-NC-SA 4.0, so non-commercial use only [11] |
| ATCOSIM | About 10 hours of simulated ATC speech | Free. Controllers only (10 non-native speakers), on close-talk headset microphones, so no pilots and no radio channel. Fine for a first fine-tune, unrepresentative for evaluation [12] |
| Own airband receiver | Live tower or approach audio | An SDR and a simple antenna; you control the recordings, so you can keep and show clips (subject to local law, below) |
| Own ADS-B receiver | Raw ADS-B at about 1 Hz, with selected altitude, heading, and altimeter setting where broadcast | An RTL-SDR dongle and a 1090 MHz antenna cost about $30–60 and remove the rate limit |
| OpenSky Network | Live and historical ADS-B state vectors | Free for non-commercial use, rate limited, 5-second resolution; barometric and geometric altitude but no selected values [9] |
| LiveATC.net | Live tower, ground, and approach audio | Personal, non-commercial use only; third-party use of the streams is prohibited [13]. Use it to pick an airport, not to build the dataset or the demo |
| METAR (aviationweather.gov) | Local altimeter setting and wind | Needed for the altitude and track corrections |

Plan for two receivers (airband and 1090 MHz) at a site with line of sight to the airport. That makes the data clean to use and the demo self-contained. Two separate SDR dongles is simplest.

You will also need to **hand-label a few hundred transmissions** from your chosen airport as a private test set: transcript, speaker, callsign, and parsed instructions. It is the most tedious part and the most valuable, because it is the only evaluation on the exact distribution you demo. Label in two passes (ASR draft, then correction) to make it faster, and have a second person label 10–20% to report agreement.

## What to measure

| Metric | Baseline | Target |
|---|---|---|
| WER on public ATC test sets | Stock Whisper; Whisper-ATC | Match or beat Whisper-ATC |
| WER on own-airport test set | Both of the above | A clear reduction; this is the honest headline |
| Callsign accuracy | Fine-tuned, no context | Fine-tuned with ADS-B biasing; compare with 82.9% in [2] |
| Speaker role accuracy | Callsign-position rule | Text model or audio features |
| Instruction extraction F1 (per type) | Rule-based parser | Best of rules, model, or hybrid |
| Linker accuracy | Exact match | Fuzzy and plausibility-weighted |
| Readback-error detection rate | None | Measured on injected errors; HAAWAII reports 81% [5] |
| False alarm rate / flag precision | None | Hand review of every live flag; HAAWAII reports about 11% false alarms [4] |
| Selected-value coverage and agreement | None | Share of clearances where the selected value matches |
| End-to-end latency | None | Under 3 s from end of transmission to readback flag |

**Readback errors are rare**, under 1% of messages [6], so the labelled set may contain none. Measure the detection rate by injecting realistic errors into real readbacks. Base the error types on the FAA taxonomy [7]: substituted or transposed digits, wrong heading or frequency, omitted elements, wrong runway. Inject at two levels:
- **Text level.** Tests the parser and checker.
- **Audio level.** Splice in digits from the same speaker elsewhere in the recording, or use TTS. Tests the whole pipeline.

Measure the false alarm rate separately: run live for several days and review every flag by hand.

Run all baselines and record the real numbers before fine-tuning. The before-and-after on your own airport is the headline of the write-up.

## Build plan

**Week 0: Setup and reading**
- Read [1] through [5]. Pick the airport and put up both receivers.
- Start recording audio and ADS-B continuously, timestamped against one NTP-synced clock. You want weeks of data by the time the pipeline is ready.
- Measure how many aircraft at your airport broadcast selected altitude, heading, and altimeter setting. That decides how central the selected-value check can be.

**Week 1: Offline speech-to-text**
- Download ATCOSIM, UWB-ATCC, and the ATCO2 1-hour test set. Fix the text normalisation.
- Measure stock Whisper and Whisper-ATC.
- Fine-tune and measure again on the public sets.

**Week 2: Labelling and parser**
- Label 300 transmissions from your airport. Measure all ASR models on them.
- Define the clearance schema. Build the rule-based parser, speaker-role rule, and callsign normaliser. Score them.

**Week 3: Live pipeline**
- Segment and transcribe the live airband stream.
- Ingest ADS-B for the same airport and link callsigns to tracks.
- Add callsign biasing and measure the gain.

**Week 4: Checker**
- Readback pairing and mismatch detection on parsed instructions.
- Selected-value and track conformance checks, with altimeter, magnetic variation, and wind corrections.
- Build the error-injection set and measure detection rate.

**Week 5: Buffer**
- Streaming, linking, and clock alignment always take longer than expected. Use this week for that, or for the model-based parser and audio speaker features if things are on time.

**Week 6: UI and live evaluation**
- Live map with transcripts and an events feed.
- Run for several days and review every flag by hand for false alarms.

**Week 7: Write-up**
- Results tables against the HAAWAII, Whisper-ATC, and Kocour numbers.
- Failure analysis.
- Three or four real caught events with audio, selected-value, and track evidence.
- A recorded demo.

## Risks

- **Too many false alarms.** Transcription errors look like readback errors. Gate flags on number-span confidence, compare parsed instructions rather than words, prefer flags confirmed by two checks, and report the false alarm rate honestly.
- **No real errors to show.** Real readback errors are rare, so a few days of live running may catch none. Non-conformance events (late compliance, go-arounds, wrong selected altitude later corrected) are more common and still make a good demo; the injected-error results carry the detection-rate claim.
- **Low selected-value coverage.** Older or general aviation aircraft often do not broadcast selected values. Measure in Week 0 and fall back to track conformance where they are missing.
- **Domain shift.** Models trained on European corpora may do much worse at a US airport. Measure early (Week 2) so there is time to label more in-domain data if needed.
- **Clocks drift.** With your own receivers on one NTP-synced clock, the problem mostly goes away. With LiveATC, the lag varies by several seconds. Estimate the offset from events visible in both: a takeoff clearance and the start of the takeoff roll.
- **Picking a hard airport.** Start with a mid-sized, single-tower airport with clear audio and good receiver coverage, not JFK.
- **Data terms.** LiveATC is personal, non-commercial use only, and third-party use of its streams is prohibited [13]. UWB-ATCC is non-commercial [11]. OpenSky is free for non-commercial use. Anything commercial needs new data.
- **Radio law.** Rules on receiving, recording, and publishing airband audio vary by country; some restrict even listening without a licence, and some forbid publishing what you hear. Check before you set up the receiver and before you publish clips.

## Stretch goals

- **Conflict prediction.** A model over ADS-B tracks that predicts positions several minutes ahead, conditioned on the parsed clearances and selected values, and warns of loss of separation. The conditioning is the interesting part; plain trajectory prediction is well studied.
- **Multi-frequency tracking.** Follow one aircraft across ground, tower, and departure.
- **On-device.** Run the whole pipeline on a small board with both SDRs and no internet feeds at all.

## Important caveat for the write-up

State clearly that this is a research demo and not a safety system. It must never be presented as something pilots or controllers could rely on.

## References

1. J. van Doorn, J. Sun, J. Hoekstra et al., "Whisper-ATC: Open Models for Air Traffic Control Automatic Speech Recognition with Accuracy," ICRAT 2024. [TU Delft repository](https://repository.tudelft.nl/record/uuid:8e02d222-5775-441d-94d2-96c26156cf43); models: [jlvdoorn/whisper-large-v3-atco2-asr](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr), [jlvdoorn/whisper-large-v3-atco2-asr-atcosim](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr-atcosim)
2. M. Kocour et al., "Boosting of Contextual Information in ASR for Air-Traffic Call-Sign Recognition," Interspeech 2021, pp. 3301–3305. [ISCA archive](https://www.isca-archive.org/interspeech_2021/kocour21_interspeech.html)
3. J. Zuluaga-Gomez et al., "BERTraffic: BERT-based Joint Speaker Role and Speaker Change Detection for Air Traffic Control Communications," SLT 2022. [arXiv:2110.05781](https://arxiv.org/abs/2110.05781)
4. H. Helmke et al., "Readback Error Detection by Automatic Speech Recognition to Increase ATM Safety," ATM Seminar 2021. [PDF](https://fit.vut.cz/research/group/speech/public/publi/2021/helmke_ATM_Seminar_2021_paper_7.pdf)
5. HAAWAII project, readback error detection results on Isavia (Iceland) recordings, SESAR Innovation Days 2022. [PDF](https://www.sesarju.eu/sites/default/files/documents/sid/2022/paper_3.pdf)
6. FAA Civil Aerospace Medical Institute studies of controller–pilot communication errors. [DOT ROSA P](https://rosap.ntl.bts.gov/view/dot/8652/dot_8652_DS1.pdf)
7. O. V. Prinzo et al., FAA Office of Aerospace Medicine reports [AM-06-25](https://libraryonline.erau.edu/online-full-text/faa-aviation-medicine-reports/AM06-25.pdf) and [AM-09-02](https://libraryonline.erau.edu/online-full-text/faa-aviation-medicine-reports/AM09-02.pdf)
8. `readsb` JSON field reference. [README-json.md](https://github.com/wiedehopf/readsb/blob/dev/README-json.md)
9. OpenSky REST API, state vectors. [Docs](https://openskynetwork.github.io/opensky-api/rest.html)
10. J. Zuluaga-Gomez et al., "ATCO2 corpus: A Large-Scale Dataset for Research on Automatic Speech Recognition and Natural Language Understanding of Air Traffic Control Communications." [arXiv:2211.04054](https://arxiv.org/abs/2211.04054); free 1-hour set: [Hugging Face](https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h)
11. UWB-ATCC, "Air Traffic Control Communication." [LINDAT](https://lindat.mff.cuni.cz/repository/items/9d93d166-c18f-459e-bc3e-d71d89f7adf2)
12. K. Hofbauer et al., "The ATCOSIM Corpus of Non-Prompted Clean Air Traffic Control Speech," LREC 2008. [TU Graz](https://www.spsc.tugraz.at/databases-and-tools/atcosim-air-traffic-control-simulation-speech-corpus.html)
13. LiveATC.net legal terms. [liveatc.net/legal](https://liveatc.net/legal)
