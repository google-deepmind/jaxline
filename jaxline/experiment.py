# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Base classes for training JAXline experiments.

Any class that implements this interface is compatible with
the JAXline Distributed Training system.
"""

import abc
import functools
import time
from typing import Dict, List, Mapping, Optional

from absl import logging

import jax
import jax.numpy as jnp
from jaxline import utils
from ml_collections import config_dict
import numpy as np


class AbstractExperiment(abc.ABC):
  """The base class for training JAXline experiments."""

  # A dict mapping attributes of this class to a name they are stored under.
  #
  # Pmapped attributes should be included in CHECKPOINT_ATTRS and will be
  # assumed to have a leading dimension corresponding to the pmapped axis when
  # saving and restoring.
  #
  # Non-pmapped attributes should be included in NON_BROADCAST_CHECKPOINT_ATTRS
  # and will be assumed to have no such leading dimension.
  CHECKPOINT_ATTRS = {}
  NON_BROADCAST_CHECKPOINT_ATTRS = {}

  @abc.abstractmethod
  def __init__(self, mode: str, init_rng: Optional[jax.Array] = None):
    """Constructs the experiment.

    Args:
      mode: A string, equivalent to FLAGS.jaxline_mode when running normally.
      init_rng: A `PRNGKey` to use for experiment initialization.
    """

    # TODO(b/205109371): Make init_rng non-optional.

  def initialize_train_step_rng(self, rng: jnp.ndarray) -> jnp.ndarray:
    """Initializes train_step_rng on devices as a suitable JAX array.

    Ensures the rng key is placed on the devices in the desired way, producing
    a JAX array of required type (i.e. `ShardedDeviceArray` or
    `GlobalDeviceArray`), partitioning and shape.

    The default implementation broadcasts the key to all local devices, forming
    a `ShardedDeviceArray` with a new leading axis. This behavior is suitable
    for the pmap-based data-parallel training.

    Args:
      rng: a single `PRNGKey`.

    Returns:
      A JAX array representing a desired configuration of rng keys on devices,
      ready to use in the training loop.
    """
    return utils.bcast_local_devices(rng)

  @abc.abstractmethod
  def step(
      self,
      *,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      writer: Optional[utils.Writer],
  ) -> Dict[str, np.ndarray]:
    """Performs a step of computation e.g. a training step.

    This function will be wrapped by `utils.kwargs_only` meaning that when
    the user re-defines this function they can take only the arguments
    they want e.g. def step(self, global_step, **unused_args).

    Args:
      global_step: A `ShardedDeviceArray` of the global step, one copy
        for each local device. The values are guaranteed to be the same across
        all local devices, it is just passed this way for consistency with
        `rng`.
      rng: A `ShardedDeviceArray` of `PRNGKey`s, one for each local device,
        and unique to the global_step. The relationship between the keys is set
        by config.random_mode_train.
      writer: An optional writer for performing additional logging (note that
        logging of the returned scalars is performed automatically by
        jaxline/train.py)

    Returns:
      A dictionary of scalar `np.array`s to be logged.
    """

  @abc.abstractmethod
  def evaluate(
      self,
      *,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      writer: Optional[utils.Writer],
  ) -> Optional[Dict[str, np.ndarray]]:

    """Performs the full evaluation of the model.

    This function will be wrapped by `utils.kwargs_only` meaning that when
    the user re-defines this function they can take only the arguments
    they want e.g. def evaluate(self, global_step, **unused_args).

    Args:
      global_step: A `ShardedDeviceArray` of the global step, one copy
        for each local device.
      rng: A `ShardedDeviceArray` of random keys, one for each local device,
        and, unlike in the step function, *independent* of the global step (i.e.
        the same array of keys is passed at every call to the function). The
        relationship between the keys is set by config.random_mode_eval.
      writer: An optional writer for performing additional logging (note that
        logging of the returned scalars is performed automatically by
        jaxline/train.py)

    Returns:
      A dictionary of scalar `np.array`s to be logged.
    """

  def should_run_step(
      self,
      global_step: int,
      config: config_dict.ConfigDict,
  ) -> bool:
    """Returns whether the step function will be run given the global_step."""
    return global_step < config.training_steps

  def train_loop(
      self,
      config: config_dict.ConfigDict,
      state,
      periodic_actions: List[utils.PeriodicAction],
      writer: Optional[utils.Writer] = None,
  ) -> None:
    """Default training loop implementation.

    Can be overridden for advanced use cases that need a different training loop
    logic, e.g. on device training loop with jax.lax.while_loop or to add custom
    periodic actions.

    Args:
      config: The config of the experiment that is being run.
      state: Checkpointed state of the experiment.
      periodic_actions: List of actions that should be called after every
        training step, for checkpointing and logging.
      writer: An optional writer to pass to the experiment step function.
    """

    @functools.partial(jax.pmap, axis_name="i")
    def next_device_state(
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        host_id: Optional[jnp.ndarray],
    ):
      """Updates device global step and rng in one pmap fn to reduce overhead."""
      global_step += 1
      step_rng, state_rng = tuple(jax.random.split(rng))
      step_rng = utils.specialize_rng_host_device(
          step_rng, host_id, axis_name="i", mode=config.random_mode_train)
      return global_step, (step_rng, state_rng)

    global_step_devices = np.broadcast_to(state.global_step - 1,
                                          [jax.local_device_count()])
    host_id_devices = utils.host_id_devices_for_rng(config.random_mode_train)
    if host_id_devices is not None:
      # Transfer to device to avoid host->device transfer on every step.
      host_id_devices = jax.pmap(lambda x: x)(host_id_devices)

    with utils.log_activity("training loop"):
      while self.should_run_step(state.global_step, config):
        with jax.profiler.StepTraceAnnotation(
            "train", step_num=state.global_step):
          # Sync global_step_devices with hosts' step and generate prng key.
          global_step_devices, (step_key, state.train_step_rng) = (
              next_device_state(global_step_devices,
                                state.train_step_rng,
                                host_id_devices))

          scalar_outputs = self.step(
              global_step=global_step_devices, rng=step_key, writer=writer)

          t = time.time()
          # Update state's global step on hosts (for checkpointing).
          # Devices' global_step_devices will be back in sync on the next
          # iteration of the while loop.
          state.global_step += 1

        for action in periodic_actions:
          action(t, state.global_step, scalar_outputs)  # pytype: disable=wrong-arg-types  # jax-ndarray

  def snapshot_state(self) -> Mapping[str, jnp.ndarray]:
    """Takes a frozen copy of the current experiment state for checkpointing.

    Returns:
      A mapping from experiment attributes to names to stored under in the
        snapshot.
    """
    snapshot_state = {}
    if not self.CHECKPOINT_ATTRS and not self.NON_BROADCAST_CHECKPOINT_ATTRS:
      logging.warning(
          "Your experiment's self.CHECKPOINT_ATTRS and "
          "self.NON_BROADCAST_CHECKPOINT_ATTRS are empty. Your job will not "
          "checkpoint any state or parameters.")
    for attr_name, chk_name in self.CHECKPOINT_ATTRS.items():
      snapshot_state[chk_name] = utils.get_first(getattr(self, attr_name))
    for attr_name, chk_name in self.NON_BROADCAST_CHECKPOINT_ATTRS.items():
      snapshot_state[chk_name] = getattr(self, attr_name)
    return snapshot_state

  def restore_from_snapshot(
      self,
      snapshot_state: Mapping[str, jnp.ndarray],
  ) -> None:
    """Restores experiment state from a snapshot.

    Args:
      snapshot_state: A mapping from experiment attributes to names they are
        stored under in the snapshot.
    """
    def clear(attributes):
      for attr_name in attributes:
        if hasattr(self, attr_name):
          delattr(self, attr_name)

    def write(attributes, broadcast=False):
      for attr_name, chk_name in attributes.items():
        value = snapshot_state[chk_name]
        if broadcast:
          value = utils.bcast_local_devices(value)
        setattr(self, attr_name, value)

    # Explicitly clear existing attributes first, this (potentially) allows
    # broadcast values to reuse previous allocations leading to reduced
    # fragmentation of device memory.
    clear(self.CHECKPOINT_ATTRS)
    clear(self.NON_BROADCAST_CHECKPOINT_ATTRS)
    write(self.CHECKPOINT_ATTRS, broadcast=True)
    write(self.NON_BROADCAST_CHECKPOINT_ATTRS)

  def on_new_best_model(self, best_state):
    """Hook to perform a custom logic when the best model is obtained.

    This method will be run before each best model checkpoint save and can
    implement any custom logic (checkpointing will still be done by jaxline).

    It will only be run if jaxline is configured to track the best model,
    i.e. if `config.best_model_eval_metric` is set.

    Args:
     best_state: Evaluator best state. Holds `best_eval_metric_value`. The state
     can also be mutated to dump additional information from the evaluator.
    """
