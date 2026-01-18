# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import checkpoint

from anemoi.training.train.tasks.base import BaseGraphModule

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection
    from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)


class GraphDiagnoser(BaseGraphModule):
    """Graph neural network diagnoser for PyTorch Lightning."""

    rollout = 0

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : BaseSchema
            Configuration object
        graph_data : HeteroData
            Dictionary of graph data for each dataset
        statistics : dict
            Training statistics
        statistics_tendencies : dict
            Tendency statistics
        data_indices : IndexCollection
            Data indices for each dataset
        metadata : dict
            Metadata
        supporting_arrays : dict
            Supporting arrays

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=None,  # statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
        )


    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Step for the Diagnoser.

        Parameters
        ----------
        batch : torch.Tensor
            Normalized batch (assumed to be already preprocessed)
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = f"Batch length not sufficient for requested multi_step!, {batch.shape[1]} !>= {self.multi_step}"
        assert batch.shape[1] >= self.multi_step, msg

        # prediction step, shape = (bs, latlon, nvar)
        y_pred = self(x)

        y = batch[
            :,
            0,  # use step 0 as target as this is an autoencoder / diagnoser!
            ...,
            self.data_indices.data.output.full,
        ]
        if y.mean() == 0:
            LOGGER.info()
            LOGGER.info("***********************")
            LOGGER.info()
            LOGGER.info(f"y_pred-statistics mean {y_pred.mean()} isnan.sum: {torch.isnan(y_pred).sum()}")
            LOGGER.info(f"y-statistics mean {y.mean()} isnan.sum: {torch.isnan(y).sum()}")

        total_loss, metrics_next, _ = checkpoint(
            self.compute_loss_metrics,
            y_pred,
            y,
            validation_mode=validation_mode,
            use_reentrant=False,
        )

        return total_loss, metrics_next, y_pred

