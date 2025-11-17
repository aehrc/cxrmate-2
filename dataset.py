from datetime import datetime
from math import e
from typing import List

import numpy as np
import torch


class CXRMate2Dataset:
    def __init__(self, dataset, history=1):
        self.dataset = dataset
        self.history = history
        self.study_id_to_index = dict(zip(self.dataset['study_id'], range(len(self.dataset)), strict=True))

    def __getitem__(self, key):
        if not isinstance(key, int):
            return self.dataset[key]

        batch = self.dataset[key]

        if 'views' not in batch:
            batch['views'] = [None] * len(batch['images'])
        
        # Set None study_datetimes to a default value:
        batch['study_datetime'] = datetime(1, 1, 1, 0, 0) if batch['study_datetime'] is None else batch['study_datetime']

        # Datetime for current study:
        batch['image_datetime'] = [batch['study_datetime'] for _ in batch['images']] 

        if self.history:
            
            if batch['prior_study_ids']:
            
                # Sort by datetime to ensure correct order:
                assert all(i is not None and not (isinstance(i, float) and np.isnan(i)) for i in batch['prior_study_datetimes'])
                prior_study_ids = [i for _, i in sorted(zip(batch['prior_study_datetimes'], batch['prior_study_ids'], strict=True))]
                prior_study_ids = prior_study_ids[-self.history:]
                
                # prior_study_datetimes = sorted(batch['prior_study_datetimes'])[-self.history:]
                prior_study_indices = [self.study_id_to_index[i] for i in prior_study_ids]
                prior_studies = [self.dataset[i] for i in prior_study_indices]

                # Datetime of prior studies:
                batch['prior_study_datetime'] = [i['study_datetime'] for i in prior_studies]
                            
                # Add prior images and their datetime:
                for study in prior_studies:

                    if 'views' not in study:
                        study['views'] = [None] * len(study['images'])

                    for image, view in zip(study['images'], study['views'], strict=True):
                        batch['images'].insert(0, image)
                        batch['views'].insert(0, view)
                        batch['image_datetime'].insert(0, study['study_datetime'])     
                
                # Prior findings and impressions:
                batch['prior_findings'] = [None if i is None else i['findings'] for i in prior_studies]     
                batch['prior_impression'] = [
                    None if i is None else i['impression'] for i in prior_studies
                ]    
            else:
                batch['prior_study_datetime'] = [None]
                batch['prior_findings'] = [None]
                batch['prior_impression'] = [None] 

        return batch

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)
    
    def __getitems__(self, keys: List):
        batch = [self.__getitem__(key) for key in keys]

        keys = set().union(*(d.keys() for d in batch))
        batch = {j: [i.setdefault(j, None) for i in batch] for j in keys}
        batch = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v for k, v in batch.items()}

        return batch
    