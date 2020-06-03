# dEFEND-web

## Introduction

dEFEND is an explainable fake news detection tool which can exploit both news contents and user comments to jointly capture explainable top-_k_ check-worthy sentences and user comments for fake news detection. It is developed by researchers from Pennsylvania State University and Arizona State University together.

If you use the code, we appreciate it if you cite an appropriate subset of the following papers:

~~~~
@inproceedings{cui2019defend,
  title={dEFEND: A System for Explainable Fake News Detection},
  author={Cui, Limeng and Shu, Kai and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2961--2964},
  year={2019}
}

@inproceedings{shu2019defend,
  title={defend: Explainable fake news detection},
  author={Shu, Kai and Cui, Limeng and Wang, Suhang and Lee, Dongwon and Liu, Huan},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={395--405},
  year={2019}
}
~~~~

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

If you are getting picking errors, run the __dos2unix.py__ file in static/saved_models, downgrade Keras further to 2.0.5, and try running it again.