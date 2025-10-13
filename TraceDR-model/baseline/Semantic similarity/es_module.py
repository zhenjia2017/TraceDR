import time
import os
from ExplaiDR.library.utils import get_logger,get_config
from es_model import ESModel

class ESModule:
    def __init__(self, config):
        self.config = config
        self.faith = self.config["faith_or_unfaith"]
        self.benchmark = self.config["benchmark"]
        self.logger = get_logger(__name__, config)
        # create model
        self.bert_model = ESModel(config)
        self.model_loaded = False

    def train(self):
        """Train the model on evidence retrieval data."""
        input_dir = self.config["path_to_intermediate_results"]
        start = time.time()
        train_path = os.path.join(self.benchmark, "train.pkl")
        dev_path = os.path.join(self.benchmark, "dev.pkl")
        self.logger.info(f"Starting training...")
        self.bert_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")
        running_time = time.time() - start
        self.logger.info(f"Training time of the scoring model is {running_time}")
        
    def test(self):
        """Test the model on evidence retrieval data."""
        input_dir = self.config["path_to_intermediate_results"]
        start = time.time()
        test_path = os.path.join(self.benchmark, "dev.pkl")
        self.logger.info(f"Starting testing...")
        self.bert_model.test(test_path)
        self.logger.info(f"Finished testing.")
        running_time = time.time() - start
        self.logger.info(f"Testing time of the scoring model is {running_time}")

    def get_top_evidences(self, query, evidences, max_evidence):
        """Run inference on a single question."""
        # load Bert model (if required)
        self._load()
        top_evidences = self.bert_model.inference_top_k(query, evidences, max_evidence)
        return top_evidences

    def _load(self):
        """Load the bert_model."""
        # only load if not already done so
        if not self.model_loaded:
            self.bert_model.load()
            self.model_loaded = True

if __name__ == "__main__":
    config = get_config('trainGCN_50_05_05.yml')
    es = ESModule(config)
    es.test()


