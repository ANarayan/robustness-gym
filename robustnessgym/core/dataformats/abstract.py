"""File containing abstract base class for datasets."""
import abc
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import cytoolz as tz
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets.arrow_dataset import DatasetInfoMixin

from robustnessgym.core.tools import recmerge

Example = Dict
Batch = Dict[str, List]


class AbstractDataset(
    abc.ABC,
    DatasetInfoMixin,
):
    """An abstract dataset class."""

    all_columns: Optional[List[str]]
    visible_columns: Optional[List[str]]
    visible_rows: Optional[np.ndarray]
    _data: Union[Dict[str, List], pd.DataFrame, pa.Table]

    def __init__(self, *args, **kwargs):
        super(AbstractDataset, self).__init__(*args, **kwargs)

    def __repr__(self):
        return f"RG{self.__class__.__name__}" f"(num_rows: {self.num_rows})"

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        # If there are columns, len of any column
        if self.column_names:
            return len(self._data[self.column_names[0]])
        return 0

    @property
    def column_names(self):
        """Column names in the dataset."""
        return self.all_columns

    @property
    def columns(self):
        """Column names in the dataset."""
        return self.column_names

    @property
    def num_rows(self):
        """Number of rows in the dataset."""
        return len(self)

    @classmethod
    def _assert_columns_all_equal_length(cls, batch: Batch):
        """Check that all columns have the same length so that the data is
        tabular."""
        assert cls._columns_all_equal_length(
            batch
        ), "All columns must have equal length."

    @classmethod
    def _columns_all_equal_length(cls, batch: Batch):
        """Check that all columns have the same length so that the data is
        tabular."""
        if len(set([len(v) for k, v in batch.items()])) == 1:
            return True
        return False

    def _check_columns_exist(self, columns: List[str]):
        """Check that every column in `columns` exists."""
        for col in columns:
            assert col in self.all_columns, f"{col} is not a valid column."

    @abc.abstractmethod
    def _set_features(self):
        """Set the features of the dataset."""
        raise NotImplementedError

    def _initialize_state(self):
        """Dataset state initialization."""
        # Show all columns by default
        self.visible_columns = self.all_columns

        # Show all rows by default
        self.visible_rows = None

        # Set the features
        self._set_features()

    @abc.abstractmethod
    def add_column(self, column: str, values: List):
        """Add a column to the dataset."""
        raise NotImplementedError

    def set_visible_rows(self, indices: Optional[Sequence]):
        """Set the visible rows in the dataset."""
        if indices is None:
            self.visible_rows = None
        else:
            if len(indices):
                assert min(indices) >= 0 and max(indices) < len(self), (
                    f"Ensure min index {min(indices)} >= 0 and "
                    f"max index {max(indices)} < {len(self)}."
                )
            self.visible_rows = np.array(indices, dtype=int)

    def reset_visible_rows(self):
        """Reset to make all rows visible."""
        self.visible_rows = None

    def set_format(self, columns: List[str]):
        """Set the dataset format.

        Only `columns` are visible after set_format is invoked.
        """
        # Check that the columns exist
        self._check_columns_exist(columns)

        # Set visible columns
        self.visible_columns = columns

    def reset_format(self):
        """Reset the dataset format.

        All columns are visible.
        """
        # All columns are visible
        self.visible_columns = self.all_columns
        # TODO(karan): all_columns should be updated if a column is added by map

    def batch(self, batch_size: int = 32, drop_last_batch: bool = False):
        """Batch the dataset.

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size

        Returns:
            batches of data
        """
        for i in range(0, len(self), batch_size):
            if drop_last_batch and i + batch_size > len(self):
                continue
            yield self[i : i + batch_size]

    @classmethod
    def from_batch(cls, batch: Batch):
        """Create an AbstractDataset from a batch."""
        return cls(batch)

    @classmethod
    def from_batches(cls, batches: List[Batch]):
        """Create an AbstractDataset from a list of batches."""
        return cls.from_batch(
            tz.merge_with(tz.compose(list, tz.concat), *batches),
        )

    def _example_or_batch_to_batch(
        self, example_or_batch: Union[Example, Batch]
    ) -> Batch:

        # Check if example_or_batch is a batch
        is_batch = all(
            [isinstance(v, List) for v in example_or_batch.values()]
        ) and self._columns_all_equal_length(example_or_batch)

        # Convert to a batch if not
        if not is_batch:
            batch = {k: [v] for k, v in example_or_batch.items()}
        else:
            batch = example_or_batch

        return batch

    @classmethod
    def _merge_batch_and_output(cls, batch: Batch, output: Batch):
        """Merge an output during .map() into a batch."""
        combined = batch
        for k in output.keys():
            if k not in batch:
                combined[k] = output[k]
            else:
                if isinstance(batch[k][0], dict) and isinstance(output[k][0], dict):
                    combined[k] = [
                        recmerge(b_i, o_i) for b_i, o_i in zip(batch[k], output[k])
                    ]
                else:
                    combined[k] = output[k]
        return combined

    @classmethod
    def _mask_batch(cls, batch: Batch, boolean_mask: List[bool]):
        """Remove elements in `batch` that are masked by `boolean_mask`."""
        return {
            k: [e for i, e in enumerate(v) if boolean_mask[i]] for k, v in batch.items()
        }

    def _inspect_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:

        # Initialize
        no_output = dict_output = bool_output = False

        # Run the function to test it
        if batched:
            if with_indices:
                output = function(self[:2], range(2))
            else:
                output = function(self[:2])

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])

        if isinstance(output, Mapping):
            dict_output = True
        elif output is None:
            no_output = True
        elif isinstance(output, bool):
            bool_output = True
        else:
            raise ValueError("function must return a dict or None.")

        return SimpleNamespace(
            dict_output=dict_output,
            no_output=no_output,
            bool_output=bool_output,
        )

    @abc.abstractmethod
    def append(
        self,
        example_or_batch: Union[Example, Batch],
    ) -> None:
        """Append a batch of data to the dataset.

        `batch` must have the same columns as the dataset (regardless of
        what columns are visible).
        """
        raise NotImplementedError
