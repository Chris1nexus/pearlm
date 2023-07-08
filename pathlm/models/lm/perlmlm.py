import torch
from torch.nn import CrossEntropyLoss
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import MaskedLMOutput


class RobertaForMaskedLMWithTypeEmb(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    SPECIAL_ID = 0
    ENTITY_ID = 1
    RELATION_ID = 2
    kg_categories = [SPECIAL_ID, ENTITY_ID, RELATION_ID]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = None

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        self.ent_mask = config.ent_mask
        self.rel_mask = config.rel_mask
        self.ent_mask = torch.FloatTensor(self.ent_mask)
        self.rel_mask = torch.FloatTensor(self.rel_mask)

        self.num_kg_types = len(RobertaForMaskedLMWithTypeEmb.kg_categories)

        # Create type embedding layer
        self.type_embeddings = torch.nn.Embedding(num_embeddings=self.num_kg_types,
                                                  embedding_dim=config.hidden_size)  # for entities, relations, and special tokens

        self.type_ids_cache = dict()
        self.type_embeds_cache = dict()

    def __init_type_embeddings(self, batch_size, num_hops):
        n_tokens = num_hops  # num_hops + 1 + num_hops + 2
        type_ids = torch.ones((batch_size, n_tokens), dtype=torch.long)

        for i in range(n_tokens):
            if i == 0 or i == n_tokens - 1:
                type_ids[:, i] = RobertaForMaskedLMWithTypeEmb.SPECIAL_ID
            elif i % 2 == 1:
                type_ids[:, i] = RobertaForMaskedLMWithTypeEmb.ENTITY_ID
            elif i % 2 == 0:
                type_ids[:, i] = RobertaForMaskedLMWithTypeEmb.RELATION_ID
        type_ids = type_ids.to(self.type_embeddings.weight.device)
        type_embeds = self.type_embeddings(type_ids)
        return type_ids, type_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        batch_size, seq_len = input_ids.shape
        k = (batch_size, seq_len)
        if k not in self.type_ids_cache:
            type_ids, type_embeds = self.__init_type_embeddings(batch_size, seq_len)

            self.type_ids_cache[k], self.type_embeds_cache[k] = type_ids, type_embeds

        type_ids, type_embeds = self.type_ids_cache[k], self.type_embeds_cache[k]

        if inputs_embeds is not None:
            inputs_embeds += type_embeds

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
