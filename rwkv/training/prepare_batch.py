"""
Prepares batches of RWKVSample data for model training and evaluation.

This module is responsible for taking raw RWKVSample objects, which represent
sequences of user review data, and transforming them into `PreparedBatch` objects.
This transformation includes:
- Adding positional and feature encodings (e.g., for IDs, day offsets).
- Padding sequences to consistent lengths within a batch.
- Implementing a complex batching strategy (`greedy_splits` or `naive_splits`)
  to handle variable-length sequences efficiently across different model submodules.
  This involves permuting and gathering data segments based on their original lengths
  to minimize padding while allowing parallel processing.
- Loading data from LMDB databases where RWKVSamples are stored.
- Providing functions to run this preparation asynchronously using multiprocessing.
"""
import math
import lmdb
import numpy as np
import torch
from rwkv.config import (
    DAY_OFFSET_ENCODE_PERIODS,
    ID_ENCODE_DIMS,
    ID_SPLIT,
    RWKV_SUBMODULES,
)
from rwkv.data_processing import ModuleData, RWKVSample
from rwkv.model.srs_model import PreparedBatch
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.utils import load_tensor


def _add_encodings(card_features, day_offsets, day_offsets_first, ids, dtype, device):
    """
    Adds ID-based and time-based (day_offset) encodings to card features.

    For each submodule ID (except user_id), generates a random encoding vector.
    For day_offsets and day_offsets_first, generates sinusoidal positional encodings
    for various periods.

    Args:
        card_features: Base tensor of card features.
        day_offsets: Tensor of day offsets for each review.
        day_offsets_first: Tensor of first day offsets for the card of each review.
        ids: Dictionary of ID tensors for each submodule.
        dtype: The data type for new tensors.
        device: The device for new tensors.

    Returns:
        A new tensor with original card_features concatenated with the new encodings.
    """
    def generate_id_encoding(submodule):
        ENCODE_DIM = ID_ENCODE_DIMS[submodule]
        return torch.randint(
            low=0,
            high=ID_SPLIT,
            size=(ENCODE_DIM,),
            device=device,
            requires_grad=False,
        ).to(dtype) - ((ID_SPLIT - 1) / 2)

    gather = [card_features]
    for submodule in RWKV_SUBMODULES:
        if submodule == "user_id":
            continue
        unique_ids = set(ids[submodule].tolist())
        # Generate a fixed random encoding for each unique ID value present in this batch
        encode = {id_val: generate_id_encoding(submodule) for id_val in unique_ids}

        encodings = []
        for id_val in ids[submodule].numpy():
            encodings.append(encode[id_val])
        gather.append(torch.stack(encodings))

    for period in DAY_OFFSET_ENCODE_PERIODS:
        # Randomly sampled baseline to improve generalization by shifting the phase
        baseline = torch.randint(low=0, high=period, size=(1,), device=device)
        f = 2 * np.pi / period
        # Day offset encodings
        encodings_sin = torch.sin(f * ((baseline + day_offsets) % period)).to(dtype)
        encodings_cos = torch.cos(f * ((baseline + day_offsets) % period)).to(dtype)
        encodings = torch.stack((encodings_sin, encodings_cos), dim=-1)
        gather.append(encodings)
        # First day offset encodings
        encodings_first_sin = torch.sin(
            f * ((baseline + day_offsets_first) % period)
        ).to(dtype)
        encodings_first_cos = torch.cos(
            f * ((baseline + day_offsets_first) % period)
        ).to(dtype)
        encodings_first = torch.stack(
            (encodings_first_sin, encodings_first_cos), dim=-1
        )
        gather.append(encodings_first)

    return torch.cat(gather, dim=-1)


def prepare(data_list: list[RWKVSample], target_len=None, seed=None) -> PreparedBatch:
    """
    Prepares a list of RWKVSample objects into a single PreparedBatch for the model.

    This is a complex function that orchestrates several steps:
    1. Sets random seed if provided.
    2. Determines `global_T`, the maximum sequence length in the batch.
    3. Calls `_add_encodings` to augment card features with ID and time encodings.
    4. Pads all augmented card features to `global_T` to create `start_tensor`.
    5. Calculates `splits` for each submodule using `greedy_splits` (or `naive_splits` implicitly
       if `target_len` is None in some code paths, though `greedy_splits` is directly called here).
       These splits define how sequences of different original lengths are grouped and padded.
    6. Iterates through submodules and their calculated splits:
        - For each segment defined by `splits` (e.g., sequences of length l to r):
            - Gathers data from `data_list` that falls into this length segment for the current submodule.
            - Permutes and pads these segments to length `r`.
            - Creates `skip` masks (for query rows) and `time_shift_select` tensors (for attention mechanics).
            - Updates `current_locs_list` to track the new positions of data elements after gathering and padding.
    7. Collects all gathered data, skip masks, and time shift selectors for each submodule.
    8. Pads global labels and label review_ths to `global_T`.
    9. Constructs and returns a `PreparedBatch` object.

    The core idea is to reorder and pad data in multiple stages, specific to each submodule's
    length distribution, to create batches that are efficient for the RWKV architecture's
    block-recurrent processing.

    Args:
        data_list: A list of RWKVSample objects.
        target_len: Target total length for sequence data, used by `greedy_splits` to balance padding and batch size.
        seed: Optional random seed for reproducibility of ID encodings and day offset baselines.

    Returns:
        A PreparedBatch object ready for model input.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed) # Numpy random operations might also be affected by seed

    with torch.no_grad():
        global_T = max([data.card_features.size(0) for data in data_list])
        data_list_t_sum = sum([data.card_features.size(0) for data in data_list])

        # Add ID and time encodings to card features
        card_features_with_ids = [
            _add_encodings(
                data.card_features, data.day_offsets, data.day_offsets_first, data.ids,
                dtype=data.card_features.dtype, device=data.card_features.device
            )
            for data in data_list
        ]
        # Pad all sequences to global_T and concatenate into a single tensor
        start_tensor = torch.cat(
            [
                torch.nn.functional.pad(
                    card_features, (0, 0, 0, global_T - card_features.size(0))
                )
                for card_features in card_features_with_ids
            ],
            dim=0,
        )

        # current_locs_list[i][j] stores the current flat index in start_tensor
        # for the j-th element of the i-th sample in data_list.
        current_locs_list = [
            i * global_T
            + torch.arange(0, data.card_features.size(0), 1, dtype=torch.long)
            for i, data in enumerate(data_list)
        ]

        # Determine how to split sequences for each submodule based on their lengths
        # factor aims to balance memory use (padding) vs. number of processing steps.
        factor = 0.9
        if target_len is None:
            # If no target_len, use a default factor for greedy_splits.
            splits = greedy_splits(data_list, factor=factor)
        else:
            # Adjust factor based on target_len to control overall batch characteristics.
            # This formula seems to aim for a certain ratio of useful data to padded data.
            splits = greedy_splits(
                data_list, factor=target_len * (1 + factor) / data_list_t_sum - 1
            )

        # --- Process each submodule based on the calculated splits ---
        # sub_gather[module_idx][split_idx] will contain the flattened, gathered, and padded data
        # for that module and split.
        sub_gather = []
        sub_skip_gather = [] # For skip masks
        sub_time_shift_gather = [] # For time shift attention mechanics
        sub_gather_lens = [] # Stores the target lengths (r values) for each split

        for submodule_name, _ in DEFAULT_ANKI_RWKV_CONFIG.modules:
            assert submodule_name in splits
            # splits[submodule_name] is a list of lengths [r1, r2, ...]
            # which define segments (0, r1], (r1, r2], ...
            module_splits_r_values = splits[submodule_name]

            # For each submodule, we will re-calculate current_locs_list based on previous submodule's processing
            # next_locs_list[data_idx][original_sequence_idx] = new_flat_index_after_this_submodule_processing
            next_locs_list = [
                np.zeros(data.card_features.size(0), dtype=np.int64)
                for data in data_list
            ]
            # all_offset tracks the current position in the flattened tensor being constructed for this submodule.
            all_offset = 0

            # Per-split data for the current submodule
            current_submodule_gather = []
            current_submodule_skip_gather = []
            current_submodule_time_shift_gather = []
            current_submodule_gather_lens = []

            for split_i in range(len(module_splits_r_values)):
                l_bound = 0 if split_i == 0 else module_splits_r_values[split_i - 1]
                r_bound = module_splits_r_values[split_i]
                current_submodule_gather_lens.append(r_bound)

                # Data to be gathered for this specific split (l_bound, r_bound]
                take_list_for_this_split = []
                skip_list_for_this_split = []
                time_shift_list_for_this_split = []

                for data_i, (data, current_locs_for_data) in enumerate(
                    zip(data_list, current_locs_list) # current_locs_list is from *previous* submodule
                ):
                    # Information about sequence lengths within this specific RWKVSample's submodule data
                    # module_data.split_len: array of unique lengths [s1, s2, ...]
                    # module_data.split_B: array of counts for each length [b1, b2, ...]
                    # module_data.from_perm: permutation to sort this sample's submodule data by length
                    module_data = data.modules[submodule_name]

                    # boundaries[k] is the starting index in module_data.from_perm
                    # for sequences of length module_data.split_len[k]
                    boundary_offset_for_sample = 0
                    boundaries = []
                    for s_l, s_b in zip(module_data.split_len, module_data.split_B):
                        boundaries.append(boundary_offset_for_sample)
                        boundary_offset_for_sample += s_l * s_b
                    boundaries.append(boundary_offset_for_sample) # Add end boundary
                    assert boundary_offset_for_sample == data.card_features.size(0)


                    # Iterate through groups of sequences of the same length within this sample's submodule data
                    for module_data_i, (data_specific_len, data_specific_B_count) in enumerate(
                        zip(module_data.split_len, module_data.split_B)
                    ):
                        # If this group of sequences (length `data_specific_len`) falls into the current target split (l_bound, r_bound]
                        if l_bound < data_specific_len and data_specific_len <= r_bound:
                            # Get the indices from from_perm for this group of sequences
                            from_slice_indices = module_data.from_perm[
                                boundaries[module_data_i] : boundaries[module_data_i + 1]
                            ]
                            # Select the actual data using these indices from current_locs_for_data
                            # (which points to locations from previous submodule's processing or start_tensor)
                            # Resulting shape: (data_specific_B_count, data_specific_len)
                            take_from_current_locs = torch.index_select(
                                current_locs_for_data, dim=0, index=from_slice_indices
                            ).view(data_specific_B_count, data_specific_len)

                            # Pad these sequences to r_bound (target length for this split)
                            # Padded with -1, which should ideally be filtered out or handled by model's masking
                            take_from_current_locs_padded = torch.nn.functional.pad(
                                take_from_current_locs,
                                (0, r_bound - data_specific_len), # Pad only on the right (sequence dim)
                                mode="constant",
                                value=-1, # Sentinel for padded values
                            )
                            take_list_for_this_split.append(take_from_current_locs_padded)

                            # Process skip masks similarly
                            skip_mask_original = torch.index_select(
                                data.skips, dim=0, index=from_slice_indices
                            ).view(data_specific_B_count, data_specific_len)
                            skip_arr_np = skip_mask_original.numpy()

                            # Calculate time_shift_select: for each position t, it's the index of the
                            # last non-skipped position <= t.
                            time_shift_select_np = np.zeros((data_specific_B_count, data_specific_len), dtype=np.int32)
                            assert (
                                skip_arr_np[0] == False # Assuming first element of any sequence is never skipped.
                            ).any(), "Cannot skip the start; otherwise we need to be careful for consecutive Trues at the start."
                            for b_idx in range(data_specific_B_count):
                                last_non_skip_t = 0
                                for t_idx in range(data_specific_len):
                                    time_shift_select_np[b_idx, t_idx] = last_non_skip_t
                                    if not skip_arr_np[b_idx, t_idx]:
                                        last_non_skip_t = t_idx

                            skip_mask_padded = torch.nn.functional.pad(
                                skip_mask_original,
                                (0, r_bound - data_specific_len),
                                mode="constant",
                                value=True, # Padded parts are skipped
                            )
                            skip_list_for_this_split.append(skip_mask_padded)

                            time_shift_select_padded = torch.nn.functional.pad(
                                torch.tensor(
                                    time_shift_select_np,
                                    dtype=torch.int32,
                                    device=skip_mask_padded.device,
                                ),
                                (0, r_bound - data_specific_len),
                                mode="constant",
                                value=0, # Padded time shifts point to beginning
                            )
                            time_shift_list_for_this_split.append(time_shift_select_padded)

                            # Update next_locs_list: for each element in the original sequence (from_slice_indices),
                            # record its new flat position (all_offset) after this submodule's processing.
                            for seq_unpadded_indices in from_slice_indices.view(
                                data_specific_B_count, data_specific_len
                            ):
                                for original_sequence_idx in seq_unpadded_indices:
                                    next_locs_list[data_i][original_sequence_idx.item()] = all_offset
                                    all_offset += 1
                                # Account for padding in the flat offset
                                all_offset += r_bound - data_specific_len

                # Concatenate all processed sequence groups for this split and flatten them
                current_submodule_gather.append(torch.cat(take_list_for_this_split, dim=0).flatten())
                current_submodule_skip_gather.append(torch.cat(skip_list_for_this_split, dim=0).flatten())
                current_submodule_time_shift_gather.append(
                    torch.cat(time_shift_list_for_this_split, dim=0).flatten().long()
                )

            # Update current_locs_list for the next submodule using the just computed next_locs_list
            current_locs_list = [torch.tensor(x, device=start_tensor.device) for x in next_locs_list] # Ensure device consistency

            sub_gather.append(current_submodule_gather)
            sub_gather_lens.append(current_submodule_gather_lens)
            sub_skip_gather.append(current_submodule_skip_gather)
            sub_time_shift_gather.append(current_submodule_time_shift_gather)

        # Pad global labels and label_review_th to global_T
        def pad_labels(labels):
            return torch.nn.functional.pad(
                labels, (0, 0, 0, global_T - labels.size(0)), mode="constant", value=0
            )

        padded_labels = torch.stack(
            list(map(lambda data: pad_labels(data.global_labels), data_list))
        )

        def pad_review_ths(labels):
            return torch.nn.functional.pad(
                labels, (0, global_T - labels.size(0)), mode="constant", value=-1 # -1 for padded review_ths
            )

        padded_label_review_th = torch.stack(
            list(map(lambda data: pad_review_ths(data.label_review_ths), data_list))
        )
        return PreparedBatch(
            num_data=len(data_list),
            start=start_tensor,
            sub_gather=sub_gather,
            sub_gather_lens=sub_gather_lens,
            skips=sub_skip_gather,
            time_shift_selects=sub_time_shift_gather,
            labels=padded_labels,
            label_review_th=padded_label_review_th,
        )


def greedy_splits(
    data_list: list[RWKVSample], factor, allowed_excess_in_one_step=20000
):
    """
    Calculates optimal split points for sequence lengths within each submodule.

    The goal is to group sequences of similar lengths together to minimize padding,
    while ensuring that the memory overhead from padding (waste) doesn't exceed
    a certain `factor` times the actual data size (used).

    Args:
        data_list: List of RWKVSamples.
        factor: Controls the trade-off between padding and number of splits.
                A higher factor allows more padding relative to used data.
        allowed_excess_in_one_step: Maximum padding "waste" allowed when adding a new
                                    group of sequences to the current split. Prevents
                                    extremely dissimilar lengths from being grouped.

    Returns:
        A dictionary mapping submodule names to a list of "r_values". These r_values
        are the upper bounds of length segments. For example, if a submodule returns
        [10, 50, 100], it means sequences are grouped into (0, 10], (10, 50], (50, 100].
    """
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        if submodule == RWKV_SUBMODULES[-1]:
            # For the last submodule, typically no complex splitting is needed,
            # just pad to the max length found in the batch for this submodule.
            longest = 0
            for data in data_list:
                module_data = data.modules[submodule]
                if module_data.split_len.size > 0: # handle cases with no data for module
                    longest = max(longest, module_data.split_len.max().item())
            splits_dict[submodule] = [longest] if longest > 0 else [0] # Ensure at least one split
            continue

        # Aggregate frequencies of each sequence length for the current submodule
        freqs = {}
        for data in data_list:
            module_data = data.modules[submodule]
            for l, b in zip(module_data.split_len, module_data.split_B):
                if l not in freqs:
                    freqs[l] = 0
                freqs[l] += b

        if not freqs: # Handle case where submodule has no data across all samples
            splits_dict[submodule] = [0]
            continue

        lens = list(reversed(sorted(freqs.keys()))) # Process from longest to shortest
        splits = []
        l_idx = 0
        while l_idx < len(lens):
            r_idx = l_idx
            # Current segment is defined by lens[l_idx] (longest sequence in this segment)
            # 'used' is data size if padded to lens[l_idx]
            # 'waste' is padding size if padded to lens[l_idx]
            used = lens[l_idx] * freqs[lens[l_idx]]
            waste = 0
            while r_idx + 1 < len(lens):
                # Try adding the next shorter group of sequences (lens[r_idx + 1])
                next_len_group = lens[r_idx + 1]
                next_freq_group = freqs[next_len_group]

                # Calculate new 'used' and 'waste' if we extend the current segment
                # to include lens[r_idx + 1], all padded to lens[l_idx].
                current_total_sequences_in_segment = sum(freqs[lens[i]] for i in range(l_idx, r_idx + 2))

                # Total data if all sequences from lens[l_idx] down to lens[r_idx+1]
                # were part of this segment (actual data, not padded size)
                next_used = sum(lens[i] * freqs[lens[i]] for i in range(l_idx, r_idx + 2))
                # Total waste if all these sequences are padded to lens[l_idx]
                next_waste = sum((lens[l_idx] - lens[i]) * freqs[lens[i]] for i in range(l_idx, r_idx + 2))

                # Extra waste introduced by just adding the lens[r_idx+1] group
                # This is (lens[l_idx] - lens[r_idx+1]) * freqs[lens[r_idx+1]]
                # This seems to be what allowed_excess_in_one_step refers to.
                # The original code had a slightly different calculation for extra_waste.
                # The condition `factor * next_used >= next_waste` is key:
                # Is (padding + data_size) / data_size <= 1 + factor?
                # (next_waste + next_used) / next_used <= 1 + factor
                # next_waste / next_used <= factor
                if (
                    factor * next_used >= next_waste and
                    (lens[l_idx] - next_len_group) * next_freq_group <= allowed_excess_in_one_step
                ):
                    # It's acceptable to merge, update r_idx
                    r_idx += 1
                else:
                    # Cannot merge, break inner loop
                    break

            splits.append(lens[l_idx]) # The r_value for this segment is the longest seq len in it
            l_idx = r_idx + 1

        splits.reverse() # Convert to ascending order of r_values
        splits_dict[submodule] = splits if splits else [0] # Ensure at least one split value

    return splits_dict


def naive_splits(data_list: list[RWKVSample]):
    """
    A simpler, less optimized way to determine submodule splits.
    It creates exponentially decreasing split points based on the longest sequence.
    (Not directly used by `prepare` if `target_len` is involved, but kept for reference/alternative).

    Args:
        data_list: List of RWKVSamples.

    Returns:
        A dictionary mapping submodule names to a list of "r_values" (split points).
    """
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        longest = 0
        for data in data_list:
            module_data = data.modules[submodule]
            if module_data.split_len.size > 0:
                longest = max(longest, module_data.split_len.max().item())

        if not longest > 0: # If no data for this module
            splits_dict[submodule] = [0]
            continue

        if submodule == RWKV_SUBMODULES[-1]:
            splits_dict[submodule] = [longest]
            continue

        splits = []
        current_max_len = longest
        while current_max_len > 0:
            splits.append(current_max_len)
            current_max_len = -1 + math.ceil(current_max_len / 1.5) # Exponentially decrease

        splits.reverse()
        splits_dict[submodule] = splits
    return splits_dict


def get_data(txn, key, device) -> RWKVSample:
    """
    Loads a single RWKVSample from an LMDB transaction given a key.

    The key itself contains user_id, start_th, end_th, and length.
    These are used to reconstruct the full keys for various tensors stored in LMDB.

    Args:
        txn: The LMDB transaction object.
        key: A tuple (user_id, start_th, end_th, length).
        device: The device to load tensors onto.

    Returns:
        An RWKVSample object.
    """
    user_id, start_th, end_th, length = key
    prefix = f"{user_id}_{start_th}-{end_th}_{length}_"
    modules = {}
    ids = {}
    for submodule in RWKV_SUBMODULES:
        module_key = prefix + submodule + "_"
        split_len = load_tensor(txn, module_key + "split_len", device=device).numpy()
        split_B = load_tensor(txn, module_key + "split_B", device=device).numpy()
        from_perm = load_tensor(txn, module_key + "from_perm", device=device)
        to_perm = load_tensor(txn, module_key + "to_perm", device=device)
        modules[submodule] = ModuleData(
            split_len=split_len, split_B=split_B, from_perm=from_perm, to_perm=to_perm
        )
        ids[submodule] = load_tensor(txn, prefix + submodule + "_id_", device=device)

    card_features = load_tensor(txn, prefix + "card_features", device=device)
    global_labels = load_tensor(txn, prefix + "global_labels", device=device)
    review_ths = load_tensor(txn, prefix + "review_ths", device=device)

    label_review_ths = load_tensor(txn, prefix + "label_review_ths", device=device)
    day_offsets = load_tensor(txn, prefix + "day_offsets", device=device)
    day_offsets_first = load_tensor(txn, prefix + "day_offsets_first", device=device)
    skips = load_tensor(txn, prefix + "skips", device=device)

    return RWKVSample(
        user_id=user_id,
        start_th=start_th,
        end_th=end_th,
        length=length,
        card_features=card_features,
        modules=modules,
        ids=ids,
        global_labels=global_labels,
        review_ths=review_ths,
        label_review_ths=label_review_ths,
        day_offsets=day_offsets,
        day_offsets_first=day_offsets_first,
        skips=skips,
    )


def prepare_data(
    lmdb_path,
    lmdb_size,
    task_queue,
    batch_queue,
    target_len=66000,
    fixed_seed=None,
):
    """
    Worker function for a multiprocessing Process to prepare data.
    Continuously fetches tasks (groups of keys) from `task_queue`,
    loads the corresponding RWKVSamples using `get_data`, prepares them
    using `prepare`, and puts the resulting `PreparedBatch` onto `batch_queue`.

    Args:
        lmdb_path: Path to the LMDB database.
        lmdb_size: Size of the LMDB database.
        task_queue: Queue to receive tasks (group_i, list_of_keys).
        batch_queue: Queue to send results (group_i, PreparedBatch).
        target_len: Target length parameter for `prepare` function.
        fixed_seed: Optional fixed seed for `prepare` function.
    """
    env = lmdb.open(lmdb_path, map_size=lmdb_size)
    with env.begin(write=False) as txn:
        while True:
            task = task_queue.get()
            if task is None: # Shutdown signal
                return

            group_i, group = task
            result = prepare(
                [get_data(txn, key, device="cpu") for key in group], # Load data on CPU
                target_len=target_len,
                seed=fixed_seed,
            )
            batch_queue.put((group_i, result))


def prepare_data_train_test(
    train_lmdb_path,
    train_lmdb_size,
    all_lmdb_path,
    all_lmdb_size,
    task_queue,
    batch_queue,
    target_len=66000,
    fixed_seed=None,
):
    """
    Similar to `prepare_data`, but handles separate LMDB databases for train and
    validation/test data. It determines which database to use based on whether
    the `group_i` in the task contains "train" or "validate".

    Args:
        train_lmdb_path: Path to the training LMDB database.
        train_lmdb_size: Size of the training LMDB database.
        all_lmdb_path: Path to the validation/test LMDB database.
        all_lmdb_size: Size of the validation/test LMDB database.
        task_queue: Queue to receive tasks.
        batch_queue: Queue to send results.
        target_len: Target length for `prepare`. Validation uses a larger default.
        fixed_seed: Optional fixed seed for `prepare`.
    """
    train_env = lmdb.open(train_lmdb_path, map_size=train_lmdb_size)
    all_env = lmdb.open(all_lmdb_path, map_size=all_lmdb_size)
    with train_env.begin(write=False) as train_txn:
        with all_env.begin(write=False) as all_txn:
            while True:
                task = task_queue.get()
                if task is None: # Shutdown signal
                    return

                group_i, group = task
                if "train" in group_i:
                    txn_to_use = train_txn
                    current_target_len = target_len
                elif "validate" in group_i:
                    txn_to_use = all_txn
                    current_target_len = 800000 # Typically validate on full sequences
                else:
                    raise ValueError(f"Task group_i '{group_i}' must contain 'train' or 'validate'")

                result = prepare(
                    [get_data(txn_to_use, key, device="cpu") for key in group],
                    target_len=current_target_len,
                    seed=fixed_seed,
                )
                batch_queue.put((group_i, result))
            def generate_id_encoding(submodule):
                ENCODE_DIM = ID_ENCODE_DIMS[submodule]
                return torch.randint(
                    low=0,
                    high=ID_SPLIT,
                    size=(ENCODE_DIM,),
                    device=card_features.device,
                    requires_grad=False,
                ).to(card_features.dtype) - ((ID_SPLIT - 1) / 2)

            gather = [card_features]
            for submodule in RWKV_SUBMODULES:
                if submodule == "user_id":
                    continue
                unique_ids = set(ids[submodule].tolist())
                encode = {id: generate_id_encoding(submodule) for id in unique_ids}

                encodings = []
                for id in ids[submodule].numpy():
                    encodings.append(encode[id])
                gather.append(torch.stack(encodings))
                # print("WARNING: zeroing out ids and rng")
                # gather.append(torch.zeros_like(torch.stack(encodings)))

            for period in DAY_OFFSET_ENCODE_PERIODS:
                # Randomly sampled baseline to improve generalization
                baseline = torch.randint(low=0, high=period, size=(1,))
                f = 2 * np.pi / period
                encodings_sin = torch.sin(f * ((baseline + day_offsets) % period)).to(
                    card_features.dtype
                )
                encodings_cos = torch.cos(f * ((baseline + day_offsets) % period)).to(
                    card_features.dtype
                )
                encodings = torch.stack((encodings_sin, encodings_cos), dim=-1)
                gather.append(encodings)
                # print("WARNING: zeroing out ids and rng")
                # gather.append(torch.zeros_like(encodings))
                encodings_first_sin = torch.sin(
                    f * ((baseline + day_offsets_first) % period)
                ).to(card_features.dtype)
                encodings_first_cos = torch.cos(
                    f * ((baseline + day_offsets_first) % period)
                ).to(card_features.dtype)
                encodings_first = torch.stack(
                    (encodings_first_sin, encodings_first_cos), dim=-1
                )
                gather.append(encodings_first)
                # print("WARNING: zeroing out ids and rng")
                # gather.append(torch.zeros_like(encodings_first))

            return torch.cat(gather, dim=-1)

        card_features_with_ids = [
            add_encodings(
                data.card_features, data.day_offsets, data.day_offsets_first, data.ids
            )
            for data in data_list
        ]
        start_tensor = torch.cat(
            [
                torch.nn.functional.pad(
                    card_features, (0, 0, 0, global_T - card_features.size(0))
                )
                for card_features in card_features_with_ids
            ],
            dim=0,
        )

        # Interpretation: the element representing a review_th of i is currently at a[i] where a[i] is a 1D tensor that holds all the data
        boundary_offset = 0
        current_locs_list = [
            i * global_T
            + torch.arange(0, data.card_features.size(0), 1, dtype=torch.long)
            for i, data in enumerate(data_list)
        ]

        # total used mem = x(1+f) where x is the sum of seq lens, f is the factor
        # at MAX and t, we use MAX*(1+t) memory
        # so f = MAX*(1+t)/x - 1
        factor = 0.9
        if target_len is None:
            splits = greedy_splits(data_list, factor=factor)
        else:
            splits = greedy_splits(
                data_list, factor=target_len * (1 + factor) / data_list_t_sum - 1
            )
        sub_gather = []
        sub_skip_gather = []
        sub_time_shift_gather = []
        sub_gather_lens = []
        for submodule_name, _ in DEFAULT_ANKI_RWKV_CONFIG.modules:
            assert submodule_name in splits
            split = splits[submodule_name]

            all_offset = 0
            next_locs_list = [
                np.zeros(data.card_features.size(0), dtype=np.int64)
                for data in data_list
            ]
            gather_lens = []
            gather = []
            skip_gather = []
            time_shift_gather = []
            for split_i in range(len(split)):
                l = 0 if split_i == 0 else split[split_i - 1]
                r = split[split_i]
                gather_lens.append(r)
                take_list = []
                skip_list = []
                time_shift_list = []

                for data_i, (data, current_locs) in enumerate(
                    zip(data_list, current_locs_list)
                ):
                    split_len = data.modules[submodule_name].split_len
                    split_B = data.modules[submodule_name].split_B
                    boundary_offset = 0
                    boundaries = []
                    for s_l, s_b in zip(split_len, split_B):
                        boundaries.append(boundary_offset)
                        boundary_offset += s_l * s_b

                    boundaries.append(boundary_offset)
                    assert boundary_offset == data.card_features.size(0)

                    module_data = data.modules[submodule_name]
                    for module_data_i, (data_split_B, data_split_len) in enumerate(
                        zip(module_data.split_B, module_data.split_len)
                    ):
                        if l < data_split_len and data_split_len <= r:
                            from_slice = module_data.from_perm[
                                boundaries[module_data_i] : boundaries[
                                    module_data_i + 1
                                ]
                            ]
                            take_from = torch.index_select(
                                current_locs, dim=0, index=from_slice
                            ).view(data_split_B, data_split_len)

                            # Maybe random instead of 0 padding to reduce collisions
                            take_from = torch.nn.functional.pad(
                                take_from,
                                (0, r - data_split_len),
                                mode="constant",
                                value=-1,
                            )
                            take_list.append(take_from)

                            skip = torch.index_select(
                                data.skips, dim=0, index=from_slice
                            ).view(data_split_B, data_split_len)
                            skip_arr = skip.numpy()
                            time_shift_select = np.zeros((data_split_B, data_split_len))
                            assert (
                                skip_arr[0] == False
                            ).any(), "Cannot skip the start; otherwise we need to be careful for consecutive Trues at the start."
                            for b in range(data_split_B):
                                last = 0
                                for t in range(data_split_len):
                                    time_shift_select[b, t] = last
                                    if not skip_arr[b, t]:
                                        last = t

                            skip = torch.nn.functional.pad(
                                skip,
                                (0, r - data_split_len),
                                mode="constant",
                                value=True,
                            )
                            skip_list.append(skip)
                            time_shift_select = torch.nn.functional.pad(
                                torch.tensor(
                                    time_shift_select,
                                    dtype=torch.int32,
                                    device=skip.device,
                                ),
                                (0, r - data_split_len),
                                mode="constant",
                                value=0,
                            )
                            time_shift_list.append(time_shift_select)

                            for seq_unpadded in from_slice.view(
                                data_split_B, data_split_len
                            ):
                                for x in seq_unpadded:
                                    next_locs_list[data_i][x] = all_offset
                                    all_offset += 1

                                all_offset += r - data_split_len
                gather.append(torch.cat(take_list, dim=0).flatten())
                skip_gather.append(torch.cat(skip_list, dim=0).flatten())
                time_shift_gather.append(
                    torch.cat(time_shift_list, dim=0).flatten().long()
                )

            sub_gather.append(gather)
            next_locs_list = [torch.tensor(x) for x in next_locs_list]
            current_locs_list = next_locs_list
            sub_gather_lens.append(gather_lens)
            sub_skip_gather.append(skip_gather)
            sub_time_shift_gather.append(time_shift_gather)

        def pad_labels(labels):
            return torch.nn.functional.pad(
                labels, (0, 0, 0, global_T - labels.size(0)), mode="constant", value=0
            )

        padded_labels = torch.stack(
            list(map(lambda data: pad_labels(data.global_labels), data_list))
        )

        def pad_review_ths(labels):
            return torch.nn.functional.pad(
                labels, (0, global_T - labels.size(0)), mode="constant", value=-1
            )

        padded_label_review_th = torch.stack(
            list(map(lambda data: pad_review_ths(data.label_review_ths), data_list))
        )
        return PreparedBatch(
            num_data=len(data_list),
            start=start_tensor,
            sub_gather=sub_gather,
            sub_gather_lens=sub_gather_lens,
            skips=sub_skip_gather,
            time_shift_selects=sub_time_shift_gather,
            labels=padded_labels,
            label_review_th=padded_label_review_th,
        )


def greedy_splits(
    data_list: list[RWKVSample], factor, allowed_excess_in_one_step=20000
):
    """'factor' puts a limit on the memory complexity.
    'allowed_excess_in_one_step' captures the notion that at some point it is better to just separate the work into sequential calls
    example: if we are given [1, 1e6] then it would be worse to pad the 1 just to fit within the same batch.
    """
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        if submodule == RWKV_SUBMODULES[-1]:
            longest = 0
            for data in data_list:
                module_data = data.modules[submodule]
                longest = max(longest, module_data.split_len.max().item())
            splits_dict[submodule] = [longest]
            continue

        freqs = {}
        for data in data_list:
            module_data = data.modules[submodule]
            for l, b in zip(module_data.split_len, module_data.split_B):
                if l not in freqs:
                    freqs[l] = 0
                freqs[l] += b

        lens = list(reversed(sorted(freqs.keys())))
        splits = []
        l = 0
        while l < len(lens):
            r = l
            used = lens[l] * freqs[lens[l]]
            waste = 0
            while r + 1 < len(lens):
                next_used = used + lens[r + 1] * freqs[lens[r + 1]]
                extra_waste = (lens[l] - lens[r + 1]) * freqs[lens[r + 1]]
                next_waste = waste + extra_waste
                if (
                    factor * next_used >= next_waste
                    and extra_waste <= allowed_excess_in_one_step
                ):
                    used = next_used
                    waste = next_waste
                    r += 1
                else:
                    break

            splits.append(lens[l])
            l = r + 1

        splits.reverse()
        splits_dict[submodule] = splits

    return splits_dict


def naive_splits(data_list: list[RWKVSample]):
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        longest = 0
        for data in data_list:
            module_data = data.modules[submodule]
            longest = max(longest, module_data.split_len.max().item())

        if submodule == RWKV_SUBMODULES[-1]:
            splits_dict[submodule] = [longest]
            continue

        splits = []
        while longest > 0:
            splits.append(longest)
            longest = -1 + math.ceil(longest / 1.5)

        splits.reverse()
        splits_dict[submodule] = splits
    return splits_dict


def get_data(txn, key, device) -> RWKVSample:
    user_id, start_th, end_th, len = key
    prefix = f"{user_id}_{start_th}-{end_th}_{len}_"
    modules = {}
    ids = {}
    for submodule in RWKV_SUBMODULES:
        module_key = prefix + submodule + "_"
        split_len = load_tensor(txn, module_key + "split_len", device=device).numpy()
        split_B = load_tensor(txn, module_key + "split_B", device=device).numpy()
        from_perm = load_tensor(txn, module_key + "from_perm", device=device)
        to_perm = load_tensor(txn, module_key + "to_perm", device=device)
        modules[submodule] = ModuleData(
            split_len=split_len, split_B=split_B, from_perm=from_perm, to_perm=to_perm
        )
        ids[submodule] = load_tensor(txn, prefix + submodule + "_id_", device=device)

    card_features = load_tensor(txn, prefix + "card_features", device=device)
    global_labels = load_tensor(txn, prefix + "global_labels", device=device)
    review_ths = load_tensor(txn, prefix + "review_ths", device=device)

    label_review_ths = load_tensor(txn, prefix + "label_review_ths", device=device)
    day_offsets = load_tensor(txn, prefix + "day_offsets", device=device)
    day_offsets_first = load_tensor(txn, prefix + "day_offsets_first", device=device)
    skips = load_tensor(txn, prefix + "skips", device=device)

    return RWKVSample(
        user_id=user_id,
        start_th=start_th,
        end_th=end_th,
        length=len,
        card_features=card_features,
        modules=modules,
        ids=ids,
        global_labels=global_labels,
        review_ths=review_ths,
        label_review_ths=label_review_ths,
        day_offsets=day_offsets,
        day_offsets_first=day_offsets_first,
        skips=skips,
    )


def prepare_data(
    lmdb_path,
    lmdb_size,
    task_queue,
    batch_queue,
    target_len=66000,
    fixed_seed=None,
):
    env = lmdb.open(lmdb_path, map_size=lmdb_size)
    with env.begin(write=False) as txn:
        while True:
            task = task_queue.get()
            if task is None:
                return

            group_i, group = task
            result = prepare(
                [get_data(txn, key, device="cpu") for key in group],
                target_len=target_len,
                seed=fixed_seed,
            )
            batch_queue.put((group_i, result))


def prepare_data_train_test(
    train_lmdb_path,
    train_lmdb_size,
    all_lmdb_path,
    all_lmdb_size,
    task_queue,
    batch_queue,
    target_len=66000,
    fixed_seed=None,
):
    train_env = lmdb.open(train_lmdb_path, map_size=train_lmdb_size)
    all_env = lmdb.open(all_lmdb_path, map_size=all_lmdb_size)
    with train_env.begin(write=False) as train_txn:
        with all_env.begin(write=False) as all_txn:
            while True:
                task = task_queue.get()
                if task is None:
                    return

                group_i, group = task
                if "train" in group_i:
                    result = prepare(
                        [get_data(train_txn, key, device="cpu") for key in group],
                        target_len=target_len,
                        seed=fixed_seed,
                    )
                elif "validate" in group_i:
                    result = prepare(
                        [get_data(all_txn, key, device="cpu") for key in group],
                        target_len=800000,
                        seed=fixed_seed,
                    )
                else:
                    raise ValueError("No key.")
                batch_queue.put((group_i, result))
