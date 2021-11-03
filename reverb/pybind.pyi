# LINT.IfChange
from typing import List, Optional, Sequence, Any, Tuple

import numpy as np


class ItemSelector: ...

class PrioritizedSelector(ItemSelector):
  def __init__(self, priority_exponent: float): ...


class FifoSelector(ItemSelector):
  def __init__(self): ...


class LifoSelector(ItemSelector):
  def __init__(self): ...


class UniformSelector(ItemSelector):
  def __init__(self): ...


class HeapSelector(ItemSelector):
  def __init__(self, min_heap: bool): ...


class TableExtension: ...


class RateLimiter:
  def __init__(self, samples_per_insert: float, min_size_to_sample: int,
               min_diff: float, max_diff: float):    ...

class Table:
  def __init__(self, name: str, sampler: ItemSelector, remover: ItemSelector,
               max_size: int, max_times_sampled: int, rate_limiter: RateLimiter,
               extensions: Sequence[TableExtension], signature: Optional[str]):
    ...
  def name(self) -> str: ...
  def can_sample(self, num_samples: int) -> bool: ...
  def can_insert(self, num_inserts: int) -> bool: ...
  def info(self) -> bytes: ...


class Writer:
  def Append(self, data): ...
  def AppendSequence(self, data): ...
  def CreateItem(self, table: str, num_timesteps: int, priority: float): ...
  def Flush(self): ...
  def Close(self, retry_on_unavailable: bool): ...


class Sampler:
  def GetNextTimestep(self) -> Tuple[List[Any], bool]: ...
  def GetNextTrajectory(self) -> List[np.ndarray]: ...


class Client:

  def __init__(
      self,
      server_name: str):
    ...

  def NewWriter(
      self,
      chunk_length: int,
      max_timesteps: int,
      delta_encoded: bool,
      max_in_flight_items: int) -> Writer:
    ...

  def NewSampler(self,
      table: str,
      max_samples: int,
      buffer_size: int) -> Sampler:
    ...

  def NewTrajectoryWriter(
      self,
      chunker_options,
      get_signature_timeout_ms: Optional[int]) -> TrajectoryWriter:
    ...

  def NewStructuredWriter(
      self,
      serialized_configs: List[bytes]) -> StructuredWriter:
    ...

  def MutatePriorities(
      self,
      table: str,
      updates: Sequence[Tuple[int, float]],
      deletes: Sequence[int]):
    ...

  def Reset(
      self,
      table: str):
    ...

  def ServerInfo(
      self,
      timeout_sec: int) -> Sequence[bytes]:
    ...

  def Checkpoint(self): ...


class Checkpointer: ...


def create_default_checkpointer(
    name: str,
    group: str,
    fallback_checkpoint_path: Optional[str]) -> Checkpointer:
  ...


class Server:
  def __init__(self, priority_tables: Sequence[Table], port: int,
               checkpointer: Optional[Checkpointer]):    ...
  def Stop(self): ...
  def Wait(self): ...



class WeakCellRef:
  def numpy(self) -> np.ndarray: ...
  @property
  def shape(self) -> List[Optional[int]]: ...
  @property
  def dtype(self) -> np.dtype: ...
  @property
  def expired(self) -> bool: ...


class ChunkerOptions: ...


class ConstantChunkerOptions(ChunkerOptions):
  def __init__(self, max_chunk_length: int, num_keep_alive_refs: int): ...


class AutoTunedChunkerOptions:
  def __init__(self, num_keep_alive_refs: int, throughput_weight: float): ...


class TrajectoryWriter:

  def Append(
      self,
      data: Sequence[Optional[Any]]) -> List[Optional[WeakCellRef]]:
    ...

  def AppendPartial(
      self,
      data: Sequence[Optional[Any]]) -> List[Optional[WeakCellRef]]:
    ...

  def CreateItem(
      self,
      table: str,
      priority: float,
      py_trajectory: Sequence[Sequence[WeakCellRef]],
      squeeze_column: Sequence[bool]):
    ...

  def Flush(
      self,
      ignore_last_num_items: int,
      timeout_ms: int):
    ...

  def EndEpisode(
      self,
      clear_buffers: bool,
      timeout_ms: Optional[int]):
    ...

  def Close(self):
    ...

  def ConfigureChunker(
      self,
      column: int,
      options: ChunkerOptions):
    ...


class StructuredWriter:
  def Append(self, data: Sequence[Optional[Any]]): ...
  def AppendPartial(self, data: Sequence[Optional[Any]]): ...
  def Flush(self, ignore_last_num_items: int, timeout_ms: int): ...
  def EndEpisode(self, clear_buffers: bool, timeout_ms: Optional[int]): ...

# LINT.ThenChange(pybind.cc)
