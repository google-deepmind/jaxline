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
"""Tests for Jaxline's train."""

import copy
from typing import Dict, Optional
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import train
from jaxline import utils
from ml_collections import config_dict
import numpy as np

_IMPROVEMENT_STEPS = [2, 5, 15, 27, 99]  # Arbitrary.
_FITNESS_METRIC_KEY = "A_GOOD_METRIC"


class DummyExperiment(experiment.AbstractExperiment):
  """An experiment whose evaluate improves at set intervals."""

  def __init__(self, mode):
    super().__init__(mode=mode)
    self.evaluate_counter = 0
    self.fitness_metric = 0
    self.init_rng = None
    self.step_rngs = []
    self.steps = 0

  def initialize_train_step_rng(self, rng: jnp.ndarray) -> jnp.ndarray:
    """Remembers the value returned from AbstractExperiment."""
    self.init_rng = super().initialize_train_step_rng(rng)
    return self.init_rng

  def step(
      self,
      *,
      global_step: jnp.ndarray,
      rng: jnp.ndarray,
      writer: Optional[utils.Writer],
  ) -> Dict[str, np.ndarray]:
    """Test implementation, counts steps and records the rngs it was given."""
    self.steps += 1
    self.step_rngs.append(rng)
    return {"loss": -1}

  def evaluate(self, *args, **kwargs) -> Optional[Dict[str, np.ndarray]]:
    """Test implementation, improves fitness metric at specified steps."""
    if self.evaluate_counter in _IMPROVEMENT_STEPS:
      self.fitness_metric += 1
    self.evaluate_counter += 1
    return {_FITNESS_METRIC_KEY: self.fitness_metric}


class DummyCheckpoint:
  """Do nothing but record when save is called."""

  def __init__(self, **kwargs):
    del kwargs  # Unused for this class.
    self._state = config_dict.ConfigDict()
    self._state_list = []
    self._checkpoint_path_int = 0
    self._global_step_int = -1

  def get_experiment_state(
      self,
      unused_ckpt_series: str,
  ) -> config_dict.ConfigDict:
    return self._state

  def save(self, unused_ckpt_series: str) -> None:
    self._state_list.append(copy.copy(self._state))

  def can_be_restored(self, ckpt_series: str) -> bool:
    return ckpt_series == "latest"

  def restore(self, unused_ckpt_series: str) -> None:
    self._global_step_int += 1
    self._state.global_step = self._global_step_int

  def restore_path(self, unused_ckpt_series) -> Optional[str]:
    """Always return something new so there"s no waiting."""
    self._checkpoint_path_int += 1
    return str(self._checkpoint_path_int)

  def wait_for_checkpointing_to_finish(self) -> None:
    """Noop, needed for API compatibility."""


class TrainTest(parameterized.TestCase):

  @parameterized.parameters(1, 4)
  def test_train_step_rng(self, num_steps: int):
    config = base_config.get_base_config()
    config.training_steps = num_steps
    checkpointer = DummyCheckpoint()
    writer = mock.create_autospec(utils.Writer, instance=True)

    train.train(DummyExperiment, config, checkpointer, writer)

    if jax.config.jax_enable_custom_prng:
      expected_rng_shape = (jax.local_device_count(),)
    else:
      expected_rng_shape = (jax.local_device_count(), 2)
    state = checkpointer.get_experiment_state("latest")

    self.assertEqual(state.global_step, num_steps)
    self.assertEqual(state.train_step_rng.shape, expected_rng_shape)

    experiment_module = state.experiment_module
    self.assertEqual(experiment_module.init_rng.shape, expected_rng_shape)
    self.assertLen(experiment_module.step_rngs, num_steps)
    for step_rng in experiment_module.step_rngs:
      self.assertEqual(step_rng.shape, expected_rng_shape)

  @parameterized.parameters(
      dict(process_id=0, checkpoint_all_hosts=False, should_checkpoint=True),
      dict(process_id=0, checkpoint_all_hosts=True, should_checkpoint=True),
      dict(process_id=0, checkpoint_all_hosts=None, should_checkpoint=True),
      dict(process_id=3, checkpoint_all_hosts=True, should_checkpoint=True),
      dict(process_id=3, checkpoint_all_hosts=False, should_checkpoint=False),
      dict(process_id=3, checkpoint_all_hosts=None, should_checkpoint=False),
  )
  def test_best_checkpoint_saves_only_at_improved_best_metrics(
      self,
      process_id: int,
      checkpoint_all_hosts: Optional[bool],
      should_checkpoint: bool,
  ):
    self.enter_context(
        mock.patch.object(jax, "process_index", new=lambda: process_id))
    config = base_config.get_base_config()
    config.best_model_eval_metric = _FITNESS_METRIC_KEY
    if checkpoint_all_hosts is not None:
      config.best_checkpoint_all_hosts = checkpoint_all_hosts
    config.training_steps = 100
    ckpt = DummyCheckpoint()
    writer = mock.Mock()
    train.evaluate(DummyExperiment, config, ckpt, writer, jaxline_mode="eval")

    if not should_checkpoint:
      self.assertEmpty(ckpt._state_list)
    else:
      # The first step will always checkpoint.
      self.assertLen(
          ckpt._state_list, len(_IMPROVEMENT_STEPS) + 1)
      checkpointed_states = [
          s.global_step for s in ckpt._state_list]
      self.assertEqual(checkpointed_states, [0] + _IMPROVEMENT_STEPS)


if __name__ == "__main__":
  absltest.main()
