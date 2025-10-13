import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
from metrics import evaluate_predictions

class FullEncoder(nn.Module):
    def __init__(self, emb_dimension=768, max_entities=100, max_evidences=50,
                 max_input_length_sr=30, max_input_length_ev=80, max_input_length_ent=60,
                 encoder_lm="BERT", encoder_linear=False):
        super(FullEncoder, self).__init__()

        # parameters
        self.emb_dimension = emb_dimension
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        self.max_input_length_sr = max_input_length_sr
        self.max_input_length_ev = max_input_length_ev
        self.max_input_length_ent = max_input_length_ent
        self.encoder_linear = encoder_linear

        # load LM
        if encoder_lm == "BERT":
            self.tokenizer = transformers.BertTokenizer.from_pretrained("hfl_chinese-roberta-wwm-ext")
            self.model = transformers.BertModel.from_pretrained("hfl_chinese-roberta-wwm-ext")
            self.sep_token = "[SEP]"
        elif encoder_lm == "DistilBERT":
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.sep_token = "[SEP]"
        elif encoder_lm == "DistilRoBERTa":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
            self.model = transformers.AutoModel.from_pretrained("distilroberta-base")
            self.sep_token = "</s>"
        else:
            raise Exception("Unknown encoder architecture specified.")

        # instantiate linear encoding layer
        if self.encoder_linear:
            self.encoder_linear_layer = torch.nn.Linear(in_features=self.max_input_length_sr, out_features=1)

        # move to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if self.encoder_linear:
                self.encoder_linear_layer = self.encoder_linear_layer.cuda()

    def encode_srs_batch(self, srs):
        """Encode all patients in the batch"""
        return self._encode(srs, max_input_length=self.max_input_length_sr)

    def encode_evidences_batch(self, evidences, *args):
        """Encode all evidences in the batch"""
        batch_size = len(evidences)
        num_evidences = len(evidences[0])

        def _prepare_input(evidence):
            return evidence["evidence_text"]

        # flatten input
        flattened_input = [
            _prepare_input(evidence)
            for i, evidences_for_inst in enumerate(evidences)
            for evidence in evidences_for_inst
        ]
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ev)
        encodings = encodings.view(batch_size, num_evidences, -1)
        return encodings

    def encode_entities_batch(self, entities, *args):
        """Encode all entities in the batch"""
        batch_size = len(entities)
        num_entities = len(entities[0])

        def _prepare_input(entity):
            entity_label = entity["label"]
            entity_type = entity["type"]
            return f"{entity_label}{self.sep_token}{entity_type}"

        # flatten input
        flattened_input = [
            _prepare_input(entity)
            for i, entities_for_inst in enumerate(entities)
            for entity in entities_for_inst
        ]
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ent)
        encodings = encodings.view(batch_size, num_entities, -1)
        return encodings

    def _encode(self, flattened_input, max_input_length):
        """Encode input string"""
        # tokenization
        tokenized_input = self.tokenizer(
            flattened_input,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            tokenized_input = tokenized_input.to(torch.device("cuda"))

        # LM encode
        outputs = self.model(**tokenized_input)
        lm_encodings = outputs.last_hidden_state

        if torch.cuda.is_available():
            lm_encodings = lm_encodings.cuda()

        # encoder linear
        if self.encoder_linear:
            encodings = self.encoder_linear_layer(lm_encodings.transpose(1, 2)).squeeze()
            encodings = F.relu(encodings)
            return encodings
        else:
            # no encoder linear (mean of embeddings)
            encodings = torch.mean(lm_encodings, dim=1)
            return encodings


class MultitaskBilinearAnswering(nn.Module):
    def __init__(self, emb_dimension=768, max_entities=100, max_evidences=50):
        super(MultitaskBilinearAnswering, self).__init__()
        self.emb_dimension = emb_dimension
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        
        # bilinear answering
        self.bilinear_answer = nn.Bilinear(emb_dimension, emb_dimension, 1)
        self.bilinear_evidence = nn.Bilinear(emb_dimension, emb_dimension, 1)
        
        # loss funciton
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, batch, train, entity_mat, sr_vec, ev_mat):
        """Arguments:
                - batch: the input batch
                - train: boolean
                - entity_mat (batch_size x num_ent x emb_dim): the entity encodings
                - sr_vec (batch_size x emb_dim): the SR vector
                - ev_mat (batch_size x num_ev x emb_dim): the evidence encodings
                - *args: other answering mechanisms require additional arguments"""
        batch_size = entity_mat.size(0)
        
        sr_vec_expanded = sr_vec.unsqueeze(1).expand(-1, self.max_entities, -1)
        answer_logits = self.bilinear_answer(entity_mat, sr_vec_expanded).squeeze(-1)
        
        sr_vec_ev_expanded = sr_vec.unsqueeze(1).expand(-1, self.max_evidences, -1)
        ev_logits = self.bilinear_evidence(ev_mat, sr_vec_ev_expanded).squeeze(-1)

        if "entity_mask" in batch:
            answer_logits = answer_logits * batch["entity_mask"]
        if "evidence_mask" in batch:
            ev_logits = ev_logits * batch["evidence_mask"]

        # calculate accuracy
        accuracies = {}
        if "entity_labels" in batch:
            answer_probs = torch.sigmoid(answer_logits)
            answer_preds = (answer_probs > 0.5).float()
            entity_accuracy = (answer_preds == batch["entity_labels"].float()).float().mean()
            accuracies["entity_accuracy"] = entity_accuracy.item()

        loss = torch.tensor(0.0, device=entity_mat.device)
        qa_metrics = []
        answer_predictions = []
        evidence_predictions = []

        if "entity_labels" in batch and "evidence_labels" in batch:
            answer_loss = self.loss_fct(answer_logits.view(-1), batch["entity_labels"].view(-1).float())
            evidence_loss = self.loss_fct(ev_logits.view(-1), batch["evidence_labels"].view(-1).float())
            
            # combined loss
            answer_weight = 0.5
            ev_weight = 0.5
            loss = answer_weight * answer_loss + ev_weight * evidence_loss

        # prediction during inference
        if not train:
            answer_predictions = self.add_ranked_answers(batch, answer_logits.unsqueeze(dim=2))
            evidence_predictions = self.add_top_evidences(batch, ev_logits.unsqueeze(dim=2), answer_logits.unsqueeze(dim=2))
            if "gold_answers" in batch:
                qa_metrics = evaluate_predictions(batch, answer_predictions, evidence_predictions)

        return {
            "loss": loss,
            "accuracies": accuracies,
            "qa_metrics": qa_metrics,
            "answer_predictions": answer_predictions,
            "ev_logits": ev_logits,
            "evidence_predictions": evidence_predictions,
        }

    def add_ranked_answers(self, batch, answer_logits):
        """Add the ranked answers (and predictions) to the output."""
        answer_predictions = []
        batch_size = answer_logits.size(0)
        # iterate through batch
        for b in range(batch_size):
            entities = batch["entities"][b]
            logits = answer_logits[b].squeeze(-1)
            
            # rank
            scores, indices = torch.sort(logits, descending=True)
            
            ranked_answers = []
            for rank, idx in enumerate(indices[:20]):  # top 20
                if idx < len(entities) and entities[idx]["id"]:
                    ranked_answers.append({
                        "answer": {"id": entities[idx]["id"], "label": entities[idx]["label"]},
                        "score": scores[rank].item(),
                        "rank": rank + 1
                    })
            
            answer_predictions.append(ranked_answers)
        
        return answer_predictions

    def add_top_evidences(self, batch, ev_logits, answer_logits):
        """Add top evidence"""
        evidence_predictions = []
        batch_size = ev_logits.size(0)
        
        for b in range(batch_size):
            evidences = batch["evidences"][b]
            logits = ev_logits[b].squeeze(-1)
            
            # sort
            scores, indices = torch.sort(logits, descending=True)
            
            top_evidences = []
            for rank, idx in enumerate(indices[:5]):  # top 5
                if idx < len(evidences) and evidences[idx].get("evidence_text"):
                    top_evidences.append({
                        "evidence": evidences[idx],
                        "score": scores[rank].item(),
                        "rank": rank + 1
                    })
            
            evidence_predictions.append(top_evidences)
        
        return evidence_predictions




class HeterogeneousGNN(nn.Module):
    """Heterogeneous Graph Neural Network"""
    def __init__(self, emb_dimension=768, num_layers=3, dropout=0.0, 
                 max_entities=100, max_evidences=50,
                 encoder_lm="BERT", encoder_linear=False):
        super(HeterogeneousGNN, self).__init__()
        
        # parameters
        self.num_layers = num_layers
        self.emb_dimension = emb_dimension
        self.dropout = dropout

        # encoder
        self.encoder = FullEncoder(
            emb_dimension=emb_dimension,
            max_entities=max_entities,
            max_evidences=max_evidences,
            encoder_lm=encoder_lm,
            encoder_linear=encoder_linear
        )

        # GNN layers
        for i in range(self.num_layers):
            # update entities
            setattr(self, f"w_ev_att_{i}", nn.Linear(emb_dimension, emb_dimension))
            setattr(self, f"w_ent_ent_{i}", nn.Linear(emb_dimension, emb_dimension))
            setattr(self, f"w_ev_ent_{i}", nn.Linear(emb_dimension, emb_dimension))

            # update evidence
            setattr(self, f"w_ent_att_{i}", nn.Linear(emb_dimension, emb_dimension))
            setattr(self, f"w_ev_ev_{i}", nn.Linear(emb_dimension, emb_dimension))
            setattr(self, f"w_ent_ev_{i}", nn.Linear(emb_dimension, emb_dimension))

        # answering module
        self.answering = MultitaskBilinearAnswering(
            emb_dimension=emb_dimension,
            max_entities=max_entities,
            max_evidences=max_evidences
        )

        # move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch, train=False):
        """Forward propagation"""
        # get data
        tsfs = batch["tsf"]
        entities = batch["entities"]
        evidences = batch["evidences"]
        ent_to_ev = batch["ent_to_ev"]
        ev_to_ent = batch["ev_to_ent"]

        # encoding
        tsf_vec = self.encoder.encode_srs_batch(tsfs)
        evidences_mat = self.encoder.encode_evidences_batch(evidences, tsfs)
        entities_mat = self.encoder.encode_entities_batch(entities, tsfs, evidences_mat, ev_to_ent, tsf_vec)

        # apply graph neural network updates
        for i in range(self.num_layers):
            ## update entities
            w_ev_att = getattr(self, f"w_ev_att_{i}")
            w_ev_ent = getattr(self, f"w_ev_ent_{i}")

            # calculate attention
            projected_evs = w_ev_att(evidences_mat)
            ev_att_scores = torch.bmm(projected_evs, tsf_vec.unsqueeze(dim=2))
            ev_att_scores = F.softmax(ev_att_scores, dim=1)
            ev_att_scores = ev_att_scores.clamp(min=1e-30, max=1e20)

            # multiply with adjacency matrix
            evidence_weights = ev_att_scores * ev_to_ent.unsqueeze(dim=0)
            evidence_weights = evidence_weights.clamp(min=1e-30, max=1e20)
            evidence_weights = evidence_weights.squeeze(dim=0).transpose(1, 2)

            # normalize
            vec = torch.sum(evidence_weights, keepdim=True, dim=2)
            vec[vec == 0] = 1
            evidence_weights = evidence_weights / vec

            # message passing: evidences -> entities
            ev_messages_ent = torch.bmm(evidence_weights, evidences_mat)
            ev_messages_ent = w_ev_ent(ev_messages_ent)

            # activation function
            entities_mat = F.relu(ev_messages_ent + entities_mat)

            ## update evidence
            w_ent_att = getattr(self, f"w_ent_att_{i}")
            w_ent_ev = getattr(self, f"w_ent_ev_{i}")

            # calculate attention
            projected_ents = w_ent_att(entities_mat)
            ent_att_scores = torch.bmm(projected_ents, tsf_vec.unsqueeze(dim=2))
            ent_att_scores = F.softmax(ent_att_scores, dim=1)
            ent_att_scores = ent_att_scores.clamp(min=1e-30, max=1e20)

            # multiply with adjacency matrix
            entity_weights = ent_att_scores * ent_to_ev.unsqueeze(dim=0)
            entity_weights = entity_weights.clamp(min=1e-30, max=1e20)
            entity_weights = entity_weights.squeeze(dim=0).transpose(1, 2)

            # normalize
            vec = torch.sum(entity_weights, keepdim=True, dim=2)
            vec[vec == 0] = 1
            entity_weights = entity_weights / vec

            # message passing: entities -> evidences
            ent_messages_ev = torch.bmm(entity_weights, entities_mat)
            ent_messages_ev = w_ent_ev(ent_messages_ev)

            # activation function
            evidences_mat = F.relu(ent_messages_ev + evidences_mat)

            # apply dropout
            entities_mat = F.dropout(entities_mat, self.dropout, training=train)
            evidences_mat = F.dropout(evidences_mat, self.dropout, training=train)

        # get answer probabilities, loss and QA metrics
        res = self.answering(batch, train, entities_mat, tsf_vec, evidences_mat)
        return res 