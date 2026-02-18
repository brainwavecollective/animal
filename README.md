# Anima*l*

```
anima /ˈæn.ɪ.mə/

the soul, especially the irrational part of the soul as distinguished from the rational mind.
```

Anima*l* extracts emotional state from raw text, and converts it into a vector that can be broadcast for various downstream purposes. Current implementation provides real-time changes in eye-color and body language for live streaming conversation for the [Reachy Mini robot](https://github.com/pollen-robotics/reachy_mini). A more significant writeup (draft-level) for the project that inspried this can be seen at:  
https://github.com/brainwavecollective/affection/blob/main/EMOTION_ENGINE.md

You can get a sense for how the engine works by running `uv run tests/smoke.py`

## Acknowledgements

This project uses the NRC Valence, Arousal, and Dominance (VAD) Lexicon (v2.1) created by Saif M. Mohammad at the National Research Council Canada. Homepage: http://saifmohammad.com/WebPages/nrc-vad.html

If you use this project in academic work, please cite:
Mohammad, Saif M. (2025). NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms. arXiv:2503.23547.

## About the Author

Daniel Ritchie is an independent technologist and founder of the Brain Wave Collective.  
[LinkedIn](https://linkedin.com/in/danielritchie123)  
[Email](mailto:daniel@brainwavecollective.ai)  
