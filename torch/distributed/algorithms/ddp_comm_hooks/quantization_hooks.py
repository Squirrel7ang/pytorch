# mypy: allow-untyped-defs
import torch
import torch.distributed as dist
from torch import nn


def _is_integer(dtype: torch.dtype):
    return dtype in [
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.short,
        torch.int32,
        torch.int,
        torch.int64,
        torch.long,
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.bool,
        torch.quint4x2,
        torch.quint2x4,
    ]


def _pack_for_bits4x2(x: torch.Tensor):
    x = x.view(-1)
    n = x.numel()
    len = (n+1) // 2

    y = torch.empty(len, dtype=x.dtype, device=x.device)
    limit = (n // 2) * 2

    x_pair = x[:limit]
    y[:limit] = (x_pair[0::2] & 0x0F) << 4 | (x_pair[1::2] & 0x0F)

    if n % 2 == 1:
        y[-1] = x[-1]
    return y


def _unpack_for_bits4x2(y: torch.Tensor, origin: torch.Tensor):
    n = origin.numel()
    x = torch.empty_like(origin)
    x[0::2] = (y[:(n//2)] & 0xF0) >> 4
    x[1::2] = (y[:(n//2)] & 0x0F)
    if n % 2 == 1:
        x[-1] = y[-1]
    return x


def _get_dtype_range(dtype):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    elif _is_integer(dtype):
        info = torch.iinfo(dtype)
    elif dtype is torch.bits4x2:
        return -8, 7
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype=}")

    return info.min, info.max


def _quantize_per_tensor_backend(x, scale, zero_point, dtype):
    d_min, d_max = _get_dtype_range(dtype)
    if dtype.is_floating_point:
        y = x / scale + zero_point
        y = torch.clamp(y, d_min, d_max).to(dtype)
    elif _is_integer(dtype):
        y = torch.round(x / scale) + zero_point
        y = torch.clamp(y, d_min, d_max).to(dtype)
    elif dtype is torch.bits4x2:
        y = torch.round(x / scale) + zero_point
        y = torch.clamp(y, d_min, d_max).to(torch.uint8)
        y = _pack_for_bits4x2(y)
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype=}")
    return y


def _dequantize_per_tensor_backend(y, scale, zero_point, dtype, origin=None):
    if dtype is torch.bits4x2:
        x = _unpack_for_bits4x2(y, origin)
        x = scale * (x.to(torch.float32) - zero_point)
    else:
        x = scale * (y.to(torch.float32) - zero_point)
    return x


def _quantize_per_channel_backend(x, scale, zero_point, dtype):
    d_min, d_max = _get_dtype_range(dtype)
    y = torch.zeros(x.size(), device=x.device)
    for i in range(x.size()[0]):
        if dtype.is_floating_point:
            y[i, :] = x[i, :] / scale[i] + zero_point[i]
        else:
            y[i, :] = torch.round(x[i, :] / scale[i]) + zero_point[i]
    y = torch.clamp(y, d_min, d_max).to(dtype)
    return y


def _dequantize_per_channel_backend(y, scale, zero_point):
    y = y.to(torch.float32).to(y.device)
    x = torch.zeros_like(y, device=y.device)
    for i in range(x.size()[0]):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list


class QuantizationState:
    __slots__ = [
        "process_group",
        "use_error_feedback",
        "error_dict",
        "dtype",
        "use_hadamard_transformation"
    ]

    def __init__(
        self,
        process_group=None,
        use_error_feedback=True,
        dtype=torch.uint8,
    ):
        self.process_group = process_group
        self.use_error_feedback = use_error_feedback
        self.error_dict: dict[int, torch.Tensor] = {}
        self.dtype = dtype

    def __getstate__(self):
        r"""
        Return a ``Dict[str, Any]`` which will be pickled and saved.

        ``process_group`` is not serializable and excluded from
        a returned state.
        """
        logger.warning(
            "NOTE: Process group is not serializable and excluded from a saved state."
        )
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot != "process_group"
        }

    def __setstate__(self, state):
        r"""
        Take a provided ``state`` and set to this ``PowerSGDState`` instance.

        ``process_group`` is set to default.
        """
        self.process_group = distributed_c10d._get_default_group()
        logger.warning(
            "NOTE: Process group will be set to a default group (i.e. the world size).\
                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded."
        )
        for slot, value in state.items():
            setattr(self, slot, value)


def quantization_pertensor_hook(
    state: QuantizationState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Apply ``torch.quantize_per_tensor`` logic to DDP using ``allgather`` protocol.

    Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    # insert error feedback
    bucket_index = bucket.index()
    total_length = tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                total_length,
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=tensor.device, dtype=dtype
            )

    # TODO: recheck if dtype is correct. The previous dtype is torch.quint8, which
    #   is also the default parameter for MinMaxObserver.
    myObserver = torch.ao.quantization.MinMaxObserver(dtype=state.dtype).to(tensor.device)
    myObserver(tensor)

    s, z = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)

    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their own ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_tensor_backend(
            tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1], state.dtype
        )
        # Store quantization error in error_dict
        if state.use_error_feedback:
            if quantized_tensor is None:
                raise AssertionError
            state.error_dict[bucket_index] = tensor - quantized_tensor
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return aggregated_dequantized_tensor / world_size

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)


def quantization_perchannel_hook(
    state: QuantizationState, bucket: dist.GradBucket, bucket_size=512
) -> torch.futures.Future[torch.Tensor]:
    """
    Apply``torch.quantize_per_channel`` logic to DDP using ``allgather`` protocol.

    Compared to per-tensor, the main motivation of per-channel is
    for considerably large tensors such as a tensor that contains 6 million
    elements quantizing per a bucket size of 512 (or 128) elements may significantly
    increase the resolution.

    It first splits ``GradBucket`` tensor into multiple chunks (channels) of ``bucket_size``
    elements. Then, workers allgather the scales and zero points of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_perchannel_hook)
    """
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    # pyrefly: ignore [missing-attribute]
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    # insert error feedback
    bucket_index = bucket.index()
    total_length = tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                total_length,
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=tensor.device, dtype=dtype
            )

    tensor_in_channels = (
        nn.functional.pad(
            input=tensor,
            pad=(0, bucket_size - len(tensor) % bucket_size),
            mode="constant",
            value=0,
        )
        .view(-1, bucket_size)
        .to(tensor.device)
    )

    # TODO: recheck if dtype is correct. The previous dtype is torch.quint8, which
    #   is also the default parameter for MinMaxObserver.
    myPerChannelObserver = torch.ao.quantization.PerChannelMinMaxObserver(
        dtype=state.dtype
    ).to(tensor.device)
    myPerChannelObserver(tensor_in_channels)

    s_ch, z_ch = myPerChannelObserver.calculate_qparams()
    s_and_z = torch.stack((s_ch, z_ch)).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their corresponding ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_channel_backend(
            tensor_in_channels,
            all_ranks_s_and_z[rank, 0, :],
            all_ranks_s_and_z[rank, 1, :],
            state.dtype,
        )
        # Store quantization error in error_dict
        if state.use_error_feedback:
            if quantized_tensor is None:
                raise AssertionError
            n = tensor.numel()
            shape = tensor.shape
            restored_quantized_tensor = quantized_tensor.flatten()[:n].view(shape)
            state.error_dict[bucket_index] = tensor - restored_quantized_tensor
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = torch.zeros_like(
            all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return (
            torch.flatten(aggregated_dequantized_tensor).to(tensor.device)[
                : tensor.size()[0]
            ]
            / world_size
        )

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)
