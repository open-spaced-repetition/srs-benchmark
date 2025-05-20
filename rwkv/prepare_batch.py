import math
import lmdb
import numpy as np
import torch
from rwkv.rwkv_config import (
    DAY_OFFSET_ENCODE_PERIODS,
    ID_ENCODE_DIMS,
    ID_SPLIT,
    RWKV_SUBMODULES,
)
from rwkv.data_processing import ModuleData, RWKVSample
from rwkv.model.srs_model import PreparedBatch
from rwkv.rwkv_config import DEFAULT_ANKI_RWKV_CONFIG
from rwkv.utils import load_tensor


def prepare(data_list: list[RWKVSample], target_len=None, seed=None) -> PreparedBatch:
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        global_T = max([data.card_features.size(0) for data in data_list])
        data_list_t_sum = sum([data.card_features.size(0) for data in data_list])

        def add_encodings(card_features, day_offsets, day_offsets_first, ids):
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
