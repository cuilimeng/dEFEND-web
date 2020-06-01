# dEFEND-web

dEFEND is an explainable fake news detection tool which can exploit both news contents and user comments to jointly capture explainable top-_k_ check-worthy sentences and user comments for fake news detection. It is developed by researchers from Pennsylvania State University and Arizona State University together.

## How to run

Note: This code is written in Python 3.

Replace "API-KEY" in __application.py__ and __templates/index.html__ with your own Rapid API Key.

```
# under static/saved_models/
unzip politifact_Defend_model.h5.zip
# back to root
python application.py
```

## Requirements

See __requirements.txt__.

## Errors

If you are getting picking errors, run the __dos2unix.py__ file in static/saved_models, and try running it again.