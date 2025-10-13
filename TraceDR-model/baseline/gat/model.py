import torch
import torch.nn.functional as F
import transformers
import numpy as np


class FullEncoder(torch.nn.Module):
    def __init__(self, emb_dimension=768, max_entities=50, max_evidences=100,
                 max_input_length_sr=128, max_input_length_ev=64, max_input_length_ent=32,
                 encoder_linear=False):
        super(FullEncoder, self).__init__()
        
        # Parameters
        self.emb_dimension = emb_dimension
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        self.max_input_length_sr = max_input_length_sr
        self.max_input_length_ev = max_input_length_ev
        self.max_input_length_ent = max_input_length_ent
        self.encoder_linear = encoder_linear

        # Load pre-trained model
        self.tokenizer = transformers.BertTokenizer.from_pretrained("../../hfl_chinese-roberta-wwm-ext")
        self.model = transformers.BertModel.from_pretrained("../../hfl_chinese-roberta-wwm-ext")
        self.sep_token = "[SEP]"
        
        if self.encoder_linear:
            self.encoder_linear_layer = torch.nn.Linear(
                in_features=self.max_input_length_sr, out_features=1
            )

        # Move to CUDA if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if self.encoder_linear:
                self.encoder_linear_layer = self.encoder_linear_layer.cuda()

    def encode_srs_batch(self, srs):
        """Encode all SRs in the batch."""
        return self._encode(srs, max_input_length=self.max_input_length_sr)

    def encode_evidences_batch(self, evidences, *args):
        """Encode all evidences in the batch."""
        batch_size = len(evidences)
        num_evidences = len(evidences[0])

        def _prepare_input(evidence):
            evidence_text = evidence["label"]
            if evidence_text is None:
                evidence_text = ""
            return evidence_text

        # Flatten input
        flattened_input = [
            _prepare_input(evidence)
            for i, evidences_for_inst in enumerate(evidences)
            for evidence in evidences_for_inst
        ]
        
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ev)
        encodings = encodings.view(batch_size, num_evidences, -1)
        return encodings

    def encode_entities_batch(self, entities, *args):
        """Encode all entities in the batch."""
        batch_size = len(entities)
        num_entities = len(entities[0])

        def _prepare_input(entity):
            return entity["name"]

        # Flatten input
        flattened_input = [
            _prepare_input(entity)
            for i, entities_for_inst in enumerate(entities)
            for entity in entities_for_inst
        ]
        
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ent)
        encodings = encodings.view(batch_size, num_entities, -1)
        return encodings

    def _encode(self, flattened_input, max_input_length):
        """Encode the given input strings."""
        # Tokenize
        tokenized_input = self.tokenizer(
            flattened_input,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )

        # Move to CUDA if available
        if torch.cuda.is_available():
            tokenized_input = tokenized_input.to(torch.device("cuda"))

        # Encode with BERT
        outputs = self.model(**tokenized_input)
        lm_encodings = outputs.last_hidden_state

        # Move to CUDA if available
        if torch.cuda.is_available():
            lm_encodings = lm_encodings.cuda()

        # Apply linear layer or mean pooling
        if self.encoder_linear:
            encodings = self.encoder_linear_layer(lm_encodings.transpose(1, 2)).squeeze()
            encodings = F.relu(encodings)
            return encodings
        else:
            encodings = torch.mean(lm_encodings, dim=1)
            return encodings


class BilinearAnswering(torch.nn.Module):
    """Bilinear answering layer for entity prediction."""
    
    def __init__(self, emb_dimension=768, dropout=0.0):
        super(BilinearAnswering, self).__init__()
        
        self.emb_dimension = emb_dimension
        self.dropout = dropout
        
        # Bilinear answering layer
        self.answer_linear_projection = torch.nn.Linear(
            in_features=self.emb_dimension, out_features=self.emb_dimension
        )
        
        # Loss function
        self.loss_fct = torch.nn.BCELoss()

    def forward(self, batch, train, entity_mat, sr_vec, *args):
        """
        Forward pass for answering.
        
        Args:
            batch: Input batch
            train: Training mode flag
            entity_mat: Entity encodings (batch_size x num_ent x emb_dim)
            sr_vec: Structured representation vector (batch_size x emb_dim)
        """
        projected_entities = self.answer_linear_projection(entity_mat)
        projected_entities = F.dropout(projected_entities, self.dropout, training=train)
        
        outputs = torch.bmm(projected_entities, sr_vec.unsqueeze(dim=2))
        logits = F.softmax(outputs, dim=1)
        mask = batch["entity_mask"]
        logits = logits.squeeze() * mask

        loss = None
        accuracies = list()
        qa_metrics = list()
        answer_predictions = list()
        
        if "entity_labels" in batch:
            loss = self.loss_fct(logits.view(-1), batch["entity_labels"].view(-1).float())

        # Compute ranked answers and QA metrics
        if not train:
            answer_predictions = self.add_ranked_answers(batch, logits.unsqueeze(dim=2))
            if "gold_answers" in batch:
                qa_metrics = self.evaluate(batch, answer_predictions)
                
        return {
            "loss": loss,
            "accuracies": accuracies,
            "qa_metrics": qa_metrics,
            "answer_predictions": answer_predictions,
        }

    def add_ranked_answers(self, batch, logits):
        """Add ranked answers to predictions."""
        answer_predictions = []
        batch_size = logits.size(0)
        
        for i in range(batch_size):
            entities = batch["entities"][i]
            scores = logits[i].squeeze()
            
            # Get valid entities (non-padded)
            valid_entities = []
            for j, entity in enumerate(entities):
                if entity["id"] != "":  # Non-padded entity
                    valid_entities.append({
                        "drug_id": entity["instruction"]["drugid"] if isinstance(entity["instruction"], dict) else entity["id"],
                        "score": scores[j].item(),
                        "name": entity["name"]
                    })
            
            # Sort by score
            valid_entities.sort(key=lambda x: x["score"], reverse=True)
            
            # Add ranks
            ranked_answers = []
            for rank, entity in enumerate(valid_entities, 1):
                ranked_answers.append({
                    "drug_id": entity["drug_id"],
                    "score": entity["score"],
                    "rank": rank,
                    "name": entity["name"]
                })
            
            answer_predictions.append({"ranked_answers": ranked_answers})
            
        return answer_predictions

    def evaluate(self, batch, answer_predictions):
        """Evaluate predictions against gold answers."""
        from metrics import precision_at_1, mrr_score, hit_at_5, calculate_metrics
        
        qa_metrics = []
        for i, pred in enumerate(answer_predictions):
            gold_answers = batch["gold_answers"][i]
            ranked_answers = pred["ranked_answers"]
            
            metrics = {
                "p_at_1": precision_at_1(ranked_answers, gold_answers),
                "mrr": mrr_score(ranked_answers, gold_answers),
                "h_at_5": hit_at_5(ranked_answers, gold_answers),
                "answer_presence": 1.0 if any(ans["drug_id"] in gold_answers for ans in ranked_answers) else 0.0
            }
            
            # Additional metrics
            additional_metrics = calculate_metrics(ranked_answers, gold_answers, i)
            metrics.update(additional_metrics)
            
            qa_metrics.append(metrics)
            
        return qa_metrics


class GAT(torch.nn.Module):
    """Graph Attention Network for drug recommendation."""
    
    def __init__(self, emb_dimension=768, num_layers=2, dropout=0.1,
                 max_entities=50, max_evidences=100, 
                 max_input_length_sr=128, max_input_length_ev=64, max_input_length_ent=32,
                 encoder_linear=False):
        super(GAT, self).__init__()
        
        # Parameters
        self.num_layers = num_layers
        self.emb_dimension = emb_dimension
        self.dropout = dropout

        # Encoder
        self.encoder = FullEncoder(
            emb_dimension=emb_dimension,
            max_entities=max_entities,
            max_evidences=max_evidences,
            max_input_length_sr=max_input_length_sr,
            max_input_length_ev=max_input_length_ev,
            max_input_length_ent=max_input_length_ent,
            encoder_linear=encoder_linear
        )

        # GNN layers
        for i in range(self.num_layers):
            # Weight matrix for all connections
            setattr(
                self,
                "w_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )
            # Attention weights
            setattr(
                self,
                "w_att_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )

        # Answering layer
        self.answering = BilinearAnswering(emb_dimension=emb_dimension, dropout=dropout)

        # Move to CUDA if available
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch, train=False):
        """Forward pass of GAT."""
        # Get data
        tsfs = batch["tsf"]
        entities = batch["entities"]
        evidences = batch["evidences"]
        
        ent_to_ev = batch["ent_to_ev"]  # batch_size x num_ent x num_ev
        ev_to_ent = batch["ev_to_ent"]  # batch_size x num_ev x num_ent

        # Encoding
        tsf_vec = self.encoder.encode_srs_batch(tsfs)
        evidences_mat = self.encoder.encode_evidences_batch(evidences, tsfs)
        entities_mat = self.encoder.encode_entities_batch(entities, tsfs, evidences_mat, ev_to_ent, tsf_vec)

        # Apply graph neural updates
        for i in range(self.num_layers):
            # Linear projection functions
            w = getattr(self, "w_" + str(i))
            w_att = getattr(self, "w_att_" + str(i))

            projected_evs = w_att(evidences_mat)
            projected_ents = w_att(entities_mat)

            # UPDATE ENTITIES
            # Compute evidence attention
            ev_att_scores = torch.bmm(projected_evs, projected_ents.transpose(1, 2))
            ev_att_scores = ev_att_scores * ev_to_ent
            ev_att_scores = F.softmax(ev_att_scores, dim=1)

            # Message passing: evidences -> entities
            ev_messages_ent = torch.bmm(ev_att_scores.transpose(1, 2), evidences_mat)
            ev_messages_ent = w(ev_messages_ent)
            
            # Updates: entities -> entities
            ent_messages_ent = w(entities_mat)
            
            # Activation function
            entities_mat = F.relu(ev_messages_ent + ent_messages_ent)

            # UPDATE EVIDENCES
            # Compute entity attention
            ent_att_scores = torch.bmm(projected_ents, projected_evs.transpose(1, 2))
            ent_att_scores = ent_att_scores * ent_to_ev
            ent_att_scores = F.softmax(ent_att_scores, dim=1)

            # Message passing: entities -> evidences
            ent_messages_ev = torch.bmm(ent_att_scores.transpose(1, 2), entities_mat)
            ent_messages_ev = w(ent_messages_ev)
            
            # Updates: evidences -> evidences
            ev_messages_ev = w(evidences_mat)
            
            # Activation function
            evidences_mat = F.relu(ev_messages_ev + ent_messages_ev)

            # Apply dropout for next layer
            entities_mat = F.dropout(entities_mat, self.dropout, training=train)
            evidences_mat = F.dropout(evidences_mat, self.dropout, training=train)

        # Get answer probabilities and metrics
        res = self.answering(batch, train, entities_mat, tsf_vec, evidences_mat)
        return res 