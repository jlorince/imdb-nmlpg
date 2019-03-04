import glob
import gzip
import json
import os
import re

import numpy
import pandas
import spacy
from tqdm import tqdm
tqdm.pandas()

print("Loading spacy...")
nlp = spacy.load('en_core_web_lg')

for filename in tqdm(glob.glob('data/amazon_products/Reviews*')):
    dataset_name = re.findall(r'(?<=reviews_)(.*)(?=_5)', filename)[0]
    base_path = f'data/amazon_products/dataframes/{dataset_name}'
    pickle_path = f'{base_path}.pkl'
    npy_path = f'{base_path}.npy'
    nlp_path = f'{base_path}.nlp.gz'
    if not (os.path.exists(pickle_path) and os.path.exists(npy_path) and os.path.exists(nlp_path)):
        print(f"Loading {filename}...")
        df = pandas.read_json(filename, lines=True)
        df['reviewRatings'] = df.helpful.progress_apply(lambda x: x[1])
        df['helpfulRatings'] = df.helpful.progress_apply(lambda x: x[0])
        del df['helpful']
        text = df['reviewText'].values
        df.to_pickle(pickle_path)
        n_docs = len(df)
        del df
        vectors = numpy.empty((n_docs, 300))
        with gzip.open(nlp_path, 'wb') as out:
            for i, doc in enumerate(tqdm(nlp.pipe(text, batch_size=10000))):
                vectors[i] = doc.vector
                out.write(f'{json.dumps(doc.print_tree())}\n'.encode('utf8'))
        numpy.save(npy_path, vectors)
