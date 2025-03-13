# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross Entropy Loss Metric"""

import evaluate
import datasets
import torch.nn.functional as F
import torch

# TODO: Add BibTeX citation
_CITATION = """
"""

# TODO: Add description of the module here
_DESCRIPTION = """\
A simple metric that calculates cross-entropy loss. Created so that I can log losses from different training tasks within a Hugging Face trainer for multi-task training.
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
    prediction_scores: the logits
    references: the labels
"""

# TODO: Define external resources urls if needed


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class cross_entropy_loss(evaluate.Metric):
    """TODO: Short description of my evaluation module."""

    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "prediction_scores": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32")
                }
            )
        )

 

    def _compute(self, prediction_scores, references):
        """Returns the scores"""

        loss = F.cross_entropy(input=torch.tensor(prediction_scores),
                            target=torch.tensor(references),
                            ignore_index=-100).item()
        return {
            "cross_entropy_loss": loss
        }