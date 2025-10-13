from rank_bm25 import BM25Okapi
import jieba

class BM25Scoring:
    def __init__(self,corpus,mapping,is_retrived):
        # with open(config["path_to_stopwords"], "r") as fp:
        #     self.stopwords = fp.read().split("\n")

        self.max_drug = 50
        self.tokenized_corpus = corpus
        self.mapping=mapping
        self.bm25_module = BM25Okapi(self.tokenized_corpus)
        self.is_retrived = is_retrived


    def get_top_evidences(self,query):
        """
        Retrieve the top-100 evidences among the retrieved ones,
        for the given AR.
        """

        def _get_evidence(tokenized_corpus, index, mapping, scores):
            #temp=dict()
            #drugs = mapping[" ".join(tokenized_corpus[index])]
            #drugs = mapping[tokenized_corpus[index]]
            if self.is_retrived:
                drug = mapping[index]
            else:
                drug = mapping[tokenized_corpus[index]]
            #temp["score"] = scores[index]
            return drug

        string = query.replace(",", " ")
        tokenized_sr = jieba.lcut(string)

        #bm25_module = BM25Okapi(self.tokenized_corpus)

        # scoring
        scores = self.bm25_module.get_scores(tokenized_sr)
        #top_doc_indices = scores.argsort()[-self.max_evidences:][::-1]
        # retrieve top-k
        ranked_indices = sorted(
            range(len(self.tokenized_corpus)), key=lambda i: scores[i], reverse=True
        )[: self.max_drug]

        scored_evidences = [
            _get_evidence(self.tokenized_corpus, index, self.mapping, scores)
            for i, index in enumerate(ranked_indices)
        ]
        return scored_evidences
