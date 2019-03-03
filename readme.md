# NMLPG Reading Group

Repository of example code, notebooks, data, etc. for the NMLPG "reading group" at Narrative Science.

## Getting started

### Dependencies
We assume you're running Python 3.6+. `requirements.txt` has most of the dependencies you'll need to run any code you find here. Just do the usual:
```
pip install -r requirements.txt
```

One extra piece of setup is to install any spacy models you'll need. Since we'll be working with word vectors, I would go ahead and install the one with the most extensive vector information:

```
python -m spacy download en_core_web_lg
```

This downloads the model to disk and is a one-time operation. You'll then be able to set up a Spacy parser by running:

```
import spacy
nlp = spacy.load('en_core_web_lg')
```

See [here](https://spacy.io/usage/models) for more details on spacy models.


If you want to run LargeVis, you'll have to install and run that separately in a *Python 2* environment following the instructions [here](https://github.com/lferry007/LargeVis). Specific examples in the relavant notebook(s) folder explain the details.

### Running notebooks
To run the notebook files, just navigate to the notebooks directory and run `jupyter notebook` from the command line. This will open the Jupyter UI in your default web browser, and you should be able to access any of the notebooks from there.


