from __future__ import annotations

import gzip
import os
import pickle
from typing import Callable, Dict, List, Optional, Union

import cytoolz as tz
import datasets
import numpy as np
import pyarrow as pa
from datasets import DatasetInfo, Features, NamedSplit

from robustnessgym.core.dataformats.abstract import AbstractDataset
from robustnessgym.core.tools import recmerge

Example = Dict
Batch = Dict[str, List]


class InMemoryDataset(AbstractDataset):
    """Class for datasets that are to be stored in memory."""

    def __init__(
        self,
        *args,
        column_names: List[str] = None,
        info: DatasetInfo = None,
        split: Optional[NamedSplit] = None,
    ):

        # Data is a dictionary of lists
        self._data = {}

        # Single argument
        if len(args) == 1:
            assert column_names is None, "Don't pass in column_names."
            # The data is passed in
            data = args[0]

            # `data` is a dictionary
            if isinstance(data, dict) and len(data):
                # Assert all columns are the same length
                self._assert_columns_all_equal_length(data)
                self._data = data

            # `data` is a list
            elif isinstance(data, list) and len(data):
                # Transpose the list of dicts to a dict of lists i.e. a batch
                data = tz.merge_with(list, *data)
                # Assert all columns are the same length
                self._assert_columns_all_equal_length(data)
                self._data = data

            # `data` is a datasets.Dataset
            elif isinstance(data, datasets.Dataset):
                self._data = data[:]
                info, split = data.info, data.split

        # No argument
        elif len(args) == 0:

            # Use column_names to setup the data dictionary
            if column_names:
                self._data = {k: [] for k in column_names}

        # Setup the DatasetInfo
        info = info.copy() if info is not None else DatasetInfo()
        AbstractDataset.__init__(self, info=info, split=split)

        # Create attributes for all columns and visible columns
        self.all_columns = list(self._data.keys())
        self.visible_columns = None

        # Create attributes for visible rows
        self.visible_rows = None

        # Initialization
        self._initialize_state()

    def _set_features(self):
        """Set the features of the dataset."""
        self.info.features = Features.from_arrow_schema(
            pa.Table.from_pydict(
                self[:1],
            ).schema
        )

    def add_column(self, column: str, values: List):
        """Add a column to the dataset."""
        assert len(values) == len(self), (
            f"`add_column` failed. "
            f"Values length {len(values)} != dataset lenth {len(self)}."
        )

        # Add the column
        self._data[column] = list(values)
        self.all_columns.append(column)

        # Set features
        self._set_features()

    def _append_to_empty_dataset(self, example_or_batch: Union[Example, Batch]) -> None:
        """Append a batch of data to the dataset when it's empty."""
        # Convert to batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # TODO(karan): what other data properties need to be in sync here
        self.all_columns = self.visible_columns = list(batch.keys())

        # Dataset is empty: create the columns and append the batch
        self._data = {k: [] for k in self.column_names}
        for k in self.column_names:
            self._data[k].extend(batch[k])

    def append(
        self,
        example_or_batch: Union[Example, Batch],
    ) -> None:
        """Append a batch of data to the dataset.

        `batch` must have the same columns as the dataset (regardless of
        what columns are visible).
        """
        if not self.column_names:
            return self._append_to_empty_dataset(example_or_batch)

        # Check that example_or_batch has the same format as the dataset
        # TODO(karan): require matching on nested features?
        columns = list(example_or_batch.keys())
        assert set(columns) == set(
            self.column_names
        ), f"Mismatched columns\nbatch: {columns}\ndataset: {self.column_names}"

        # Convert to a batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # Append to the dataset
        for k in self.column_names:
            self._data[k].extend(batch[k])

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        if (
            isinstance(index, int)
            or isinstance(index, slice)
            or isinstance(index, np.int)
        ):
            # int or slice index => standard list slicing
            return {k: self._data[k][index] for k in self.visible_columns}
        elif isinstance(index, str):
            # str index => column selection
            if index in self.column_names:
                if self.visible_rows is not None:
                    return [self._data[index][i] for i in self.visible_rows]
                return self._data[index]
            raise AttributeError(f"Column {index} does not exist.")
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return {k: [self._data[k][i] for i in index] for k in self.visible_columns}
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def select_columns(self, columns: List[str]) -> Batch:
        """Select a subset of columns."""
        for col in columns:
            assert col in self._data
        return tz.keyfilter(lambda k: k in columns, self._data)

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        # keep_in_memory: bool = False,
        #             # load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        #             # writer_batch_size: Optional[int] = 1000,
        #             # features: Optional[Features] = None,
        #             # disable_nullable: bool = False,
        #             # fn_kwargs: Optional[dict] = None,
        #             # num_proc: Optional[int] = None,
        #             # suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        #             # new_fingerprint: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> Optional[InMemoryDataset]:

        # Just return if the function is None
        if function is None:
            return None

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Set the format
        if input_columns:
            previous_format = self.visible_columns
            self.set_format(input_columns)

        # Get some information about the function
        function_properties = self._inspect_function(function, with_indices, batched)
        update_dataset = function_properties.dict_output

        # Return if `self` has no examples
        if not len(self):
            return self if update_dataset else None

        # Map returns a new dataset if the function returns a dict
        if update_dataset:
            new_dataset = InMemoryDataset()

        # Run the map
        if batched:
            for i, batch in enumerate(self.batch(batch_size, drop_last_batch)):
                output = (
                    function(
                        batch,
                        range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                    )
                    if with_indices
                    else function(batch)
                )

                if update_dataset:
                    # TODO(karan): check that this has the correct behavior
                    output = self._merge_batch_and_output(batch, output)
                    # output = recmerge(batch, output, merge_sequences=True)
                    new_dataset.append(output)

        else:
            for i, example in enumerate(self):
                output = function(example, i) if with_indices else function(example)

                if update_dataset:
                    # TODO(karan): check that this has the correct behavior
                    output = recmerge(example, output)
                    new_dataset.append(output)

        # Reset the format
        if input_columns:
            self.set_format(previous_format)

        if update_dataset:
            new_dataset._set_features()
            return new_dataset
        return None

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        # keep_in_memory: bool = False,
        # load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        # writer_batch_size: Optional[int] = 1000,
        # fn_kwargs: Optional[dict] = None,
        # num_proc: Optional[int] = None,
        # suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        # new_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        # Set the function if it's None
        if function is None:
            function = lambda *args, **kwargs: True

        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Set the format
        if input_columns:
            previous_format = self.visible_columns
            self.set_format(input_columns)

        # Get some information about the function
        # TODO(karan): extend to handle batched functions
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=False,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Run the filter
        indices = []
        for i, example in enumerate(self):
            output = (
                function(
                    example,
                    i,
                )
                if with_indices
                else function(example)
            )
            assert isinstance(output, bool), "function must return boolean."

            # Add in the index
            if output:
                indices.append(i)

        # Reset the format, to set visible columns for the filter
        self.reset_format()

        # Filter returns a new dataset
        new_dataset = InMemoryDataset()
        if indices:
            new_dataset = InMemoryDataset.from_batch(self[indices])

        # Set the format back to what it was before the filter was applied
        if input_columns:
            self.set_format(previous_format)

        return new_dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_data", "all_columns", "_info", "_split"}

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def __getstate__(self) -> Dict:
        """Get the internal state of the dataset."""
        state = {key: getattr(self, key) for key in self._state_keys()}
        self._assert_state_keys(state)
        return state

    def __setstate__(self, state: Dict) -> None:
        """Set the internal state of the dataset."""
        if not isinstance(state, dict):
            raise ValueError(
                f"`state` must be a dictionary containing " f"{self._state_keys()}."
            )

        self._assert_state_keys(state)

        for key in self._state_keys():
            setattr(self, key, state[key])

        # Do some initialization
        self._initialize_state()

    @classmethod
    def load_from_disk(cls, path: str) -> InMemoryDataset:
        """Load the in-memory dataset from disk."""

        with gzip.open(os.path.join(path, "data.gz")) as f:
            dataset = pickle.load(f)
        # # Empty state dict
        # state = {}
        #
        # # Load the data
        # with gzip.open(os.path.join(path, "data.gz")) as f:
        #     state['_data'] = pickle.load(f)
        #
        # # Load the metadata
        # metadata = json.load(
        #     open(os.path.join(path, "metadata.json"))
        # )
        #
        # # Merge the metadata into the state
        # state = {**state, **metadata}

        # Create an empty `InMemoryDataset` and set its state
        # dataset = cls()
        # dataset.__setstate__(state)

        return dataset

    def save_to_disk(self, path: str):
        """Save the in-memory dataset to disk."""
        # Create all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Store the data in a compressed format
        with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
            pickle.dump(self, f)

        # # Get the dataset state
        # state = self.__getstate__()
        #
        # # Store the data in a compressed format
        # with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
        #     pickle.dump(state['_data'], f)
        #
        # # Store the metadata
        # json.dump(
        #     {k: v for k, v in state.items() if k != '_data'},
        #     open(os.path.join(path, "metadata.json"), 'w'),
        # )
