import torch
from typing import Dict, Optional

from ..Metric.WandBMetricOrchestrator import WandBMetricOrchestrator


class TrackerBalancedLoss:

    def __init__(
        self,
        loss_lkp:Dict,
        mo:Optional[WandBMetricOrchestrator]=None,
        name:Optional[str] = ""
        ):
        """Class for handling MultiLoss outputs

        Args:
            loss_lkp (Dict[str:torch.nn.modules.loss]): Dictionary containing 
            the required losses and lookup values. The lookup values should
            match those passed to the call method in act and pred parameters
        """
        self.loss_lkp = loss_lkp
        self.mo = mo
        self.__step = 1
        self.name = name

    def __call__(
        self, 
        pred:Dict[str, torch.Tensor],
        act:Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        """Evaluates the input values against the loss functions specified in
        the init.

        Args:
            act (Dict[str:torch.Tensor]): Dictionary of the
            form {"name_of_loss": actual_values}
            The keys should match the keys provided in the loss_lkp parameter,
            specified in the init
            pred (Dict[str:torch.Tensor]): Dictionary of the
            form {"name_of_loss": predicted_values}
            The keys should match the keys provided in the loss_lkp parameter,
            specified in the init

        Returns:
            Dict[str:Any]: Dictionary of evaluated results. The keys will match
            those provided in the multi_loss parameter
        """
        loss = 0
        _metric_value_dict = {}
        if "mu" in pred and "logvar" in pred:
            kl = self.loss_lkp["kl"](pred["mu"], pred["logvar"])
            recon = self.loss_lkp["images"](pred["images"], act["images"])
            loss += kl
            loss += recon
        else:
            for key in self.loss_lkp.keys():
                _loss = self.loss_lkp[key](pred[key], act[key])
                _metric_value_dict[f"{key}_{self.name}_loss"] = {
                    "label":f"step_{self.__step}",
                    "value":_loss
                }
                loss += _loss
        if self.mo is not None:
            self.mo.update_metrics(metric_value_dict=_metric_value_dict)
        out_loss = torch.mean(loss)
        self.__step += 1
        return out_loss
