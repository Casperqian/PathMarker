# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from dataclasses import dataclass, field as field_ptr_behaviour, fields, is_dataclass
from typing import Any, get_args, get_origin, List, Union, Tuple

import torch

from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)

from .sam3_image_dataset import Datapoint
from sam3.train.mixed_supervision import DENSE_MASK, supervision_mode_id


MyTensor = Union[torch.Tensor, List[Any]]


def convert_my_tensors(obj):
    def is_optional_field(field) -> bool:
        return get_origin(field) is Union and type(None) in get_args(field)

    for field in fields(obj):
        if is_dataclass(getattr(obj, field.name)):
            convert_my_tensors(getattr(obj, field.name))
            continue

        field_type = field.type
        if is_optional_field(field.type):
            field_type = Union[get_args(field.type)[:-1]]  # Get the Optional field type

        if field_type != MyTensor or getattr(obj, field.name) is None:
            continue

        elif len(getattr(obj, field.name)) and isinstance(
            getattr(obj, field.name)[0], torch.Tensor
        ):
            stack_dim = 0
            if field.name in [
                "input_boxes",
                "input_boxes_label",
            ]:
                stack_dim = 1
            setattr(
                obj,
                field.name,
                torch.stack(getattr(obj, field.name), dim=stack_dim).to(
                    getattr(obj, field.name + "__type")
                ),
            )
        else:
            setattr(
                obj,
                field.name,
                torch.as_tensor(
                    getattr(obj, field.name), dtype=getattr(obj, field.name + "__type")
                ),
            )
    return obj


def packed_to_padded_naive(boxes_packed, num_boxes, fill_value=0):
    """
    Convert a packed tensor of bounding boxes to a padded tensor of bounding
    boxes. Naive implementation using a loop.

    Inputs:
    - boxes_packed: Tensor of shape (N_1 + ... + N_B, 4)
    - num_boxes: Tensor of shape (B,) where num_boxes[i] = N_i

    Returns:
    - boxes_padded: Tensor of shape (B, N_max, 4) where N_max = max_i N_i
    """
    B = num_boxes.shape[0]
    Ns = num_boxes.tolist()

    boxes_padded = boxes_packed.new_zeros(B, max(Ns), *boxes_packed.shape[1:])
    if fill_value != 0:
        boxes_padded[...] = fill_value
    prev_idx = 0
    for i in range(B):
        next_idx = prev_idx + Ns[i]
        boxes_padded[i, : Ns[i]] = boxes_packed[prev_idx:next_idx]
        prev_idx = next_idx
    return boxes_padded


def pad_tensor_list_to_longest(
    tensors: List[torch.Tensor], dim=0, pad_val=0
) -> List[torch.Tensor]:
    # Edits the list in-place
    if not tensors:
        return tensors
    pad_len = max(t.shape[dim] for t in tensors)
    for i in range(len(tensors)):
        n_dims = len(tensors[i].shape)
        n_right_dims = (n_dims - 1) - (n_dims + dim) % n_dims
        n_pad = pad_len - tensors[i].shape[dim]
        pad_tuple = tuple([0] * 2 * n_right_dims + [0, n_pad])
        tensors[i] = torch.nn.functional.pad(tensors[i], pad_tuple, value=pad_val)
    return tensors


def collate_fn_api_with_chunking(
    batch,
    num_chunks,
    dict_key,
    with_seg_masks=False,
    input_points_embedding_dim=257,
    repeats: int = 0,
    load_image_in_fp16: bool = False,
):
    assert num_chunks >= 1, "num_chunks must be >= 1"

    # split the batch into num_chunks chunks
    batch_chunks = [batch[i::num_chunks] for i in range(num_chunks)]

    # collate each chunk
    collated_chunks = [
        collate_fn_api(
            chunk,
            dict_key,
            with_seg_masks,
            input_points_embedding_dim,
            repeats,
            # ptr_behaviour,
            load_image_in_fp16,
        )
        for chunk in batch_chunks
    ]
    return collated_chunks


def collate_fn_api(
    batch: List[Datapoint],
    dict_key,
    with_seg_masks=False,
    input_points_embedding_dim=257,
    repeats: int = 0,
    load_image_in_fp16: bool = False,
):
    # img_batch = torch.stack(sum([[img.data for img in v.images] for v in batch], []))
    img_batch = []
    text_batch = []
    raw_images = None

    num_stages = (
        max(q.query_processing_order for data in batch for q in data.find_queries) + 1
    )

    stages = [
        FindStage(
            img_ids=[],
            text_ids=[],
            input_boxes=[],
            input_boxes_label=[],
            input_boxes_mask=[],
            input_points=[],
            input_points_mask=[],
            object_ids=[],
        )
        for _ in range(num_stages)
    ]
    find_targets = [
        BatchedFindTarget(
            num_boxes=[],
            boxes=[],
            boxes_padded=[],
            target_points=[],
            target_points_padded=[],
            valid_boxes=[],
            valid_boxes_padded=[],
            valid_points=[],
            valid_points_padded=[],
            supervision_modes=[],
            valid_dense_target=[],
            is_exhaustive=[],
            segments=[],
            semantic_segments=[],
            is_valid_segment=[],
            repeated_boxes=[],
            object_ids=[],
            object_ids_padded=[],
        )
        for _ in range(num_stages)
    ]
    find_metadatas = [
        BatchedInferenceMetadata(
            coco_image_id=[],
            original_size=[],
            object_id=[],
            frame_index=[],
            original_image_id=[],
            original_category_id=[],
            is_conditioning_only=[],
        )
        for _ in range(num_stages)
    ]

    offset_img_id = 0
    offset_query_id = [0 for _ in range(num_stages)]
    label_batch = []
    image_path_batch = []
    patch_info_batch = []
    for i, data in enumerate(batch):
        img_batch.extend([img.data for img in data.images])

        if data.raw_images is not None:
            if raw_images is None:
                raw_images = []
            raw_images.extend(data.raw_images)
        
        label_batch.append(getattr(data, "label", None))
        image_path_batch.append(getattr(data, "image_path", None))
        patch_info_batch.append(getattr(data, "patch_info", None))

        # Conversion of query_ids indexing in a datapoint to query_ids indexing in a stage
        datapoint_query_id_2_stage_query_id = []
        for q in data.find_queries:
            stage_id = q.query_processing_order
            datapoint_query_id_2_stage_query_id.append(offset_query_id[stage_id])
            offset_query_id[stage_id] += 1

        for j, q in enumerate(data.find_queries):
            stage_id = q.query_processing_order
            stages[stage_id].img_ids.append(q.image_id + offset_img_id)
            if q.query_text not in text_batch:
                text_batch.append(q.query_text)
            stages[stage_id].text_ids.append(text_batch.index(q.query_text))

            assert (
                q.inference_metadata is not None
            ), "inference_metadata must be provided when FindQueryLoaded is created."
            for f in fields(q.inference_metadata):
                getattr(find_metadatas[stage_id], f.name).append(
                    getattr(q.inference_metadata, f.name)
                )

            if q.input_bbox is not None:
                assert q.input_bbox.numel() % 4 == 0
                assert q.input_bbox_label is not None
                nb_boxes = q.input_bbox.numel() // 4
                assert len(q.input_bbox_label) == nb_boxes
                stages[stage_id].input_boxes.append(q.input_bbox.view(nb_boxes, 4))
                stages[stage_id].input_boxes_label.append(
                    q.input_bbox_label.view(nb_boxes)
                )
                stages[stage_id].input_boxes_mask.append(
                    torch.zeros(nb_boxes, dtype=torch.bool)
                )
            else:
                stages[stage_id].input_boxes.append(torch.zeros(0, 4))
                stages[stage_id].input_boxes_label.append(
                    torch.zeros(0, dtype=torch.bool)
                )
                stages[stage_id].input_boxes_mask.append(
                    torch.ones(0, dtype=torch.bool)
                )

            if q.input_points is not None:
                # print(f"Found input points for stage {stage_id}, query {j} in datapoint {i}, with shape {q.input_points.shape}")
                stages[stage_id].input_points.append(
                    q.input_points.squeeze(0)  # Strip a trivial batch index
                )
                # All masks will be padded up to the longest length
                # with 1s before final conversion to batchd tensors
                stages[stage_id].input_points_mask.append(
                    torch.zeros(q.input_points.shape[1])
                )
            else:
                # print(f"No input points for stage {stage_id}, query {j} in datapoint {i}, adding empty tensor with shape (0, {input_points_embedding_dim})")
                # stages[stage_id].input_points.append(
                #     torch.empty(0, input_points_embedding_dim)
                # )
                stages[stage_id].input_points.append(torch.empty(0, 3))
                stages[stage_id].input_points_mask.append(torch.empty(0))

            current_out_boxes = []
            current_out_object_ids = []
            current_target_points = []
            current_valid_boxes = []
            current_valid_points = []
            # Set the object ids referred to by this query
            stages[stage_id].object_ids.append(q.object_ids_output)
            for object_id in q.object_ids_output:
                target_object = data.images[q.image_id].objects[object_id]
                current_out_boxes.append(target_object.bbox)
                current_out_object_ids.append(object_id)
                point = target_object.target_point
                current_target_points.append(
                    point if point is not None else torch.zeros(2, dtype=torch.float32)
                )
                current_valid_boxes.append(bool(target_object.valid_box))
                current_valid_points.append(bool(target_object.valid_point))
            find_targets[stage_id].boxes.extend(current_out_boxes)
            find_targets[stage_id].object_ids.extend(current_out_object_ids)
            find_targets[stage_id].target_points.extend(current_target_points)
            find_targets[stage_id].valid_boxes.extend(current_valid_boxes)
            find_targets[stage_id].valid_points.extend(current_valid_points)
            if repeats > 0:
                for _ in range(repeats):
                    find_targets[stage_id].repeated_boxes.extend(current_out_boxes)
            find_targets[stage_id].num_boxes.append(len(current_out_boxes))
            find_targets[stage_id].is_exhaustive.append(q.is_exhaustive)
            supervision_mode = supervision_mode_id(q.supervision_mode)
            find_targets[stage_id].supervision_modes.append(supervision_mode)
            find_targets[stage_id].valid_dense_target.append(
                supervision_mode == DENSE_MASK
            )

            if with_seg_masks:
                current_seg_mask = []
                current_is_valid_segment = []
                for object_id in q.object_ids_output:
                    seg_mask = data.images[q.image_id].objects[object_id].segment
                    if seg_mask is not None:
                        current_seg_mask.append(seg_mask)
                        current_is_valid_segment.append(1)
                    else:
                        dummy_mask = torch.zeros(
                            data.images[q.image_id].data.shape[-2:], dtype=torch.bool
                        )
                        current_seg_mask.append(dummy_mask)
                        current_is_valid_segment.append(0)
                find_targets[stage_id].segments.extend(current_seg_mask)
                find_targets[stage_id].is_valid_segment.extend(current_is_valid_segment)
            else:
                # We are not loading segmentation masks
                find_targets[stage_id].segments = None
                find_targets[stage_id].is_valid_segment = None

            if q.semantic_target is not None:
                find_targets[stage_id].semantic_segments.append(q.semantic_target)

        offset_img_id += len(data.images)

    # Pad input points to equal sequence lengths
    for i in range(len(stages)):
        stages[i].input_points = pad_tensor_list_to_longest(
            stages[i].input_points, dim=0, pad_val=0
        )
        # Masked-out regions indicated by 1s.
        stages[i].input_points_mask = pad_tensor_list_to_longest(
            stages[i].input_points_mask, dim=0, pad_val=1
        )

    # Pad input boxes to equal sequence lengths
    for i in range(len(stages)):
        stages[i].input_boxes = pad_tensor_list_to_longest(
            stages[i].input_boxes, dim=0, pad_val=0
        )
        stages[i].input_boxes_label = pad_tensor_list_to_longest(
            stages[i].input_boxes_label, dim=0, pad_val=0
        )
        # Masked-out regions indicated by 1s.
        stages[i].input_boxes_mask = pad_tensor_list_to_longest(
            stages[i].input_boxes_mask, dim=0, pad_val=1
        )

    # Convert to tensors
    for i in range(len(stages)):
        try:
            stages[i] = convert_my_tensors(stages[i])
        except Exception as e:
            print(f"Error converting stage {i} to tensors {stages[i]=}: {e}")
        
        find_targets[i] = convert_my_tensors(find_targets[i])
        find_metadatas[i] = convert_my_tensors(find_metadatas[i])
        # get padded representation for the boxes
        find_targets[i].boxes_padded = packed_to_padded_naive(
            find_targets[i].boxes.view(-1, 4), find_targets[i].num_boxes
        )
        find_targets[i].object_ids_padded = packed_to_padded_naive(
            find_targets[i].object_ids, find_targets[i].num_boxes, fill_value=-1
        )
        find_targets[i].target_points_padded = packed_to_padded_naive(
            find_targets[i].target_points.view(-1, 2), find_targets[i].num_boxes
        )
        find_targets[i].valid_boxes_padded = packed_to_padded_naive(
            find_targets[i].valid_boxes, find_targets[i].num_boxes
        )
        find_targets[i].valid_points_padded = packed_to_padded_naive(
            find_targets[i].valid_points, find_targets[i].num_boxes
        )

    # Finalize the image batch
    # check sizes
    for img in img_batch[1:]:
        assert img.shape == img_batch[0].shape, "All images must have the same size"
    image_batch = torch.stack(img_batch)
    if load_image_in_fp16:
        # Optionally, cast the image tensors to fp16, which helps save GPU memory on
        # long videos with thousands of frames (where image tensors could be several GBs)
        image_batch = image_batch.half()

    return {
        "label_batch": label_batch,
        "img_path_batch": image_path_batch,
        "patch_info_batch": patch_info_batch,
        dict_key: BatchedDatapoint(
            img_batch=image_batch,
            find_text_batch=text_batch,
            find_inputs=stages,
            find_targets=find_targets,
            find_metadatas=find_metadatas,
            raw_images=raw_images,
        ),
    }


from typing import Tuple, Any
from dataclasses import fields, is_dataclass
import torch


def _select_by_query_mask(x: Any, qmask: torch.Tensor):
    """
    Select elements corresponding to qmask along the 'query dimension'.
    Supports:
      - Tensor with shape [Q, ...]  -> x[qmask]
      - Tensor with shape [L, Q, ...] -> x[:, qmask]
      - list length Q -> [v for v, keep in zip(x, qmask.tolist()) if keep]
      - None -> None
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        Q = int(qmask.shape[0])
        if x.dim() >= 1 and x.shape[0] == Q:
            return x[qmask]
        if x.dim() >= 2 and x.shape[1] == Q:
            return x[:, qmask]
        return x
    if isinstance(x, list):
        qm = qmask.tolist()
        if len(x) == len(qm):
            return [v for v, keep in zip(x, qm) if keep]
        return x
    return x


def _dataclass_select_by_query_mask(obj: Any, qmask: torch.Tensor):
    """
    For a dataclass, apply _select_by_query_mask on each field.
    """
    if obj is None:
        return None
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass, got {type(obj)}")

    kwargs = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        kwargs[f.name] = _select_by_query_mask(v, qmask)
    return obj.__class__(**kwargs)


def split_batched_datapoint(
    bd: "BatchedDatapoint",
) -> Tuple["BatchedDatapoint", "BatchedDatapoint"]:
    """
    Split bd where images are interleaved as:
      [dp0_v0, dp0_v1, dp1_v0, dp1_v1, ..., dp{B-1}_v0, dp{B-1}_v1]

    Returns:
      bd_raw: contains all v0 images (size B)
      bd_aug: contains all v1 images (size B)

    Also filters/remaps stage.img_ids accordingly:
      - raw keeps even img_ids; new_img_id = old_img_id // 2
      - aug keeps odd  img_ids; new_img_id = old_img_id // 2
    """
    num_images = int(bd.img_batch.shape[0])
    if num_images % 2 != 0:
        raise ValueError(f"Expected even num_images (2*B). Got {num_images}.")

    img_batch_raw = bd.img_batch[0::2]  # dp*_v0
    img_batch_aug = bd.img_batch[1::2]  # dp*_v1

    def _filter_stage(stage: "FindStage", keep_even: bool):
        img_ids = stage.img_ids  # [Q]
        qmask = (img_ids % 2 == 0) if keep_even else (img_ids % 2 == 1)

        # stage 内部字段可能是 [Q,...] 或 [L,Q,...]，用统一选择器处理
        new_stage = FindStage(
            img_ids=(img_ids[qmask] // 2),
            text_ids=_select_by_query_mask(stage.text_ids, qmask),
            input_boxes=_select_by_query_mask(stage.input_boxes, qmask),
            input_boxes_label=_select_by_query_mask(stage.input_boxes_label, qmask),
            input_boxes_mask=_select_by_query_mask(stage.input_boxes_mask, qmask),
            input_points=_select_by_query_mask(stage.input_points, qmask),
            input_points_mask=_select_by_query_mask(stage.input_points_mask, qmask),
            object_ids=_select_by_query_mask(stage.object_ids, qmask),
        )
        return new_stage, qmask

    bd_raw_find_inputs, bd_aug_find_inputs = [], []
    bd_raw_find_targets, bd_aug_find_targets = [], []
    bd_raw_find_metadatas, bd_aug_find_metadatas = [], []

    for stage, targets, metas in zip(bd.find_inputs, bd.find_targets, bd.find_metadatas):
        stage_raw, mask_raw = _filter_stage(stage, keep_even=True)
        stage_aug, mask_aug = _filter_stage(stage, keep_even=False)

        bd_raw_find_inputs.append(stage_raw)
        bd_aug_find_inputs.append(stage_aug)

        bd_raw_find_targets.append(_dataclass_select_by_query_mask(targets, mask_raw))
        bd_aug_find_targets.append(_dataclass_select_by_query_mask(targets, mask_aug))

        bd_raw_find_metadatas.append(_dataclass_select_by_query_mask(metas, mask_raw))
        bd_aug_find_metadatas.append(_dataclass_select_by_query_mask(metas, mask_aug))

    raw_images = getattr(bd, "raw_images", None)
    if raw_images is not None:
        try:
            raw_images_raw = raw_images[0::2]
            raw_images_aug = raw_images[1::2]
        except Exception:
            raw_images_raw = raw_images
            raw_images_aug = raw_images
    else:
        raw_images_raw = None
        raw_images_aug = None

    bd_raw = BatchedDatapoint(
        img_batch=img_batch_raw,
        find_text_batch=bd.find_text_batch,
        find_inputs=bd_raw_find_inputs,
        find_targets=bd_raw_find_targets,
        find_metadatas=bd_raw_find_metadatas,
        raw_images=raw_images_raw,
    )
    bd_aug = BatchedDatapoint(
        img_batch=img_batch_aug,
        find_text_batch=bd.find_text_batch,
        find_inputs=bd_aug_find_inputs,
        find_targets=bd_aug_find_targets,
        find_metadatas=bd_aug_find_metadatas,
        raw_images=raw_images_aug,
    )
    return bd_raw, bd_aug
