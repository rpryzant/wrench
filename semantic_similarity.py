"""
tried 

        import semantic_similarity
        scorer = semantic_similarity.Scorer(self.dataset, vocab)  
        scores = scorer.train(texts, labels)

        for idx, cid, x in name_rulefn_score:
            print(id2tok[idx], cid, x, scorer.score(id2tok[idx]))


"""


from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn import preprocessing
from torch.functional import _return_inverse
from tqdm import tqdm
from scipy.stats import entropy

DATA_INFO = {
    'chemprot': {
        'description': 'classifying the relationship between chemicals and proteins',
        'labels': {
            0: 'part of',
            1: 'regulator',
            2: 'upregulator',
            3: 'downregulator',
            4: 'agonist',
            5: 'antagonist',
            6: 'modulator',
            7: 'cofactor',
            8: 'substrate/product',
            9: 'not'
        },
        'full_labels': {
            0: 'the chemical is part of the protein',
            1: 'the chemical is a regulator of the protein',
            2: 'the chemical is an upregulator of the protein',
            3: 'the chemical is a downregulator of the protein',
            4: 'the chemical is an agonist of the protein',
            5: 'the chemical is an antagonist of the protein',
            6: 'the chemical is a modulator of the protein',
            7: 'the chemical is a cofactor for the protein',
            8: 'the chemical is a substrate/product for the protein',
            9: 'not'
        },
    }


}


class Scorer:
    def __init__(self, dataset, word2idx):
        # best for semantic similarity
        self.model = SentenceTransformer('all-mpnet-base-v2')
        # TODO HERE!!

        self.label2emb = {}
        inputs = [
            # f"Description: '{DATA_INFO[dataset]['description']}' Label: '{label_name}'"
            label_name
            for label_idx, label_name in DATA_INFO[dataset]['full_labels'].items()
        ]
        self.embs = self.model.encode(inputs, show_progress_bar=False, convert_to_numpy=True)
        self.embs = preprocessing.normalize(self.embs, norm='l2', axis=1)

        idx2word = {idx: word for word, idx in word2idx.items()}
        word2wpcs = {word: self.model.tokenize([word])['input_ids'].tolist()[0][1:-1] for word in word2idx}
        self.wpcs2word = {tuple(wpc): word for word, wpc in word2wpcs.items()}
        self.word2embs = defaultdict(list)

    def train(self, texts, labels):

        for text, lab in tqdm(list(zip(texts, labels))):
            toks = self.model.tokenize([text])['input_ids'].tolist()[0]
            emb = self.model.encode([text], show_progress_bar=False, output_value='token_embeddings')[0].cpu().numpy()
            emb = preprocessing.normalize(emb, norm='l2', axis=1)

            for wpc, word in self.wpcs2word.items():
                idx = contains_sublist(toks, list(wpc))
                if idx is None:
                    continue

                tok_embs = emb[idx: idx + len(wpc)]
                tok_emb = np.mean(tok_embs, axis=0)

                # sims = np.dot(tok_emb, self.embs.T)
                # if np.max(sims) > 0.4:
                #     print(word)
                #     print(text)
                #     print(np.argmax(sims), lab)

                self.word2embs[word].append(tok_emb)

    def score(self, word):
        if word not in self.word2embs:
            return -1

        # print(word)
        # print(np.array(self.word2embs[word]).shape)
        mean_emb = np.mean(self.word2embs[word], axis=0)
        word_sims = np.dot(mean_emb, self.embs.T)
        return word_sims




def contains_sublist(test_list, sublist):
    res = None
    for idx in range(len(test_list) - len(sublist) + 1):
            if test_list[idx : idx + len(sublist)] == sublist:
                res = idx 
                break
    return res
