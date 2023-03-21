from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch


class PadCollate:
    def __init__(self, pad_token_id: int, entity_pad_token_id: int,):
        self._pad_token_id = pad_token_id
        self._entity_pad_token_id = entity_pad_token_id

    def get_pad_id(self, key: str) -> int:
        if key.endswith("input_ids"):
            return self._pad_token_id
        elif key.endswith("attention_mask"):
            return 0
        elif key.endswith("entity_ids"):
            return self._entity_pad_token_id
        elif key.endswith("entity_attention_mask"):
            return 0
        elif key.endswith("entity_position_ids"):
            return -1
        else:
            assert False, f"Unknown key {key}"

    def __call__(self, batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        from torch.utils.data.dataloader import default_collate  # type: ignore
        from irtools.pad import pad_to

        keys = {k for k, v in (batch[0])[0].items() if isinstance(v, torch.Tensor)}
        unalign = {k for k in keys if len(set((x[0])[k].size() for x in batch)) > 1}
        sizes = {k: np.max([(x[0])[k].size() for x in batch], axis=0) for k in unalign}

        query_encoded_output = [
            {
                k: pad_to(v, sizes[k], self.get_pad_id(k)) if k in unalign else v
                for k, v in x[0].items()
            }
            for x in batch
        ]
        keys = {k for k, v in (batch[0])[1].items() if isinstance(v, torch.Tensor)}
        unalign = {k for k in keys if len(set((x[1])[k].size() for x in batch)) > 1}
        sizes = {k: np.max([(x[1])[k].size() for x in batch], axis=0) for k in unalign}
        doc_encoded_output = [
            {
                k: pad_to(v, sizes[k], self.get_pad_id(k)) if k in unalign else v
                for k, v in x[1].items()
                }
            for x in batch
            ]
        queryID = [
            x[2]
            for x in batch
            ]
        docset =[
            x[3]
            for x in batch
        ]
        relevance_grades = [
            x[4]
            for x in batch
        ]

        query_collated: Dict[str, torch.Tensor] = default_collate(query_encoded_output)
        doc_collated: Dict[str, torch.Tensor] = default_collate(doc_encoded_output)

        # if self._index_key:
        #     assert self._index_key not in collated, f"Key {self._index_key} exists"
        #     collated[self._index_key] = torch.arange(len(batch))
        return query_collated, doc_collated, queryID, docset, relevance_grades
