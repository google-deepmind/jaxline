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
"""Tests for jaxline's utils."""

import functools
import itertools as it
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import jax
import jax.numpy as jnp
from jaxline import experiment
from jaxline import utils
from ml_collections import config_dict
import numpy as np


def _num_unique_keys(keys):
  return len(np.unique(jax.random.key_data(keys), axis=0))


class PyPrefetchTest(absltest.TestCase):

  def testEmpty(self):
    self.assertEqual(list(utils.py_prefetch(lambda: ())), [])

  def testBaseCase(self):
    self.assertEqual(list(utils.py_prefetch(lambda: range(100))),
                     list(range(100)))

  def testBadFunction(self):

    def _bad_function():
      raise ValueError

    iterable = utils.py_prefetch(_bad_function)
    with self.assertRaises(ValueError):
      next(iterable)

  def testBadFunctionIteration(self):

    def _bad_iterable():
      yield 1
      raise ValueError

    iterable = utils.py_prefetch(_bad_iterable)
    self.assertEqual(next(iterable), 1)
    with self.assertRaises(ValueError):
      next(iterable)


class TreePsumTest(absltest.TestCase):

  def testBaseCase(self):
    # pick leaf objects with leading dimension one as these tests will
    # be run on a single device.
    data = {"a": jnp.array([1]), "b": jnp.array([2])}
    data_summed = jax.pmap(
        lambda x: utils.tree_psum(x, axis_name="i"), axis_name="i")(data)
    self.assertEqual(data_summed, data)

  def testEmpty(self):
    data = {"a": jnp.array([]), "b": jnp.array([])}
    with self.assertRaises(ZeroDivisionError):
      jax.pmap(lambda x: utils.tree_psum(x, axis_name="i"), axis_name="i")(data)

  def testSingleLeafTree(self):
    data = jnp.array([1])
    data_summed = jax.pmap(
        lambda x: utils.tree_psum(x, axis_name="i"), axis_name="i")(data)
    self.assertEqual(data_summed, data)

  def testNotNumpy(self):
    data = [1]
    with self.assertRaises(ValueError):
      jax.pmap(lambda x: utils.tree_psum(x, axis_name="i"), axis_name="i")(data)

  def testNumDevicesMismatch(self):
    data = jnp.array([1, 2])  # assumes 2 devices but we only have 1
    with self.assertRaises(ValueError):
      jax.pmap(lambda x: utils.tree_psum(x, axis_name="i"), axis_name="i")(data)

  def testNoPmapWrapper(self):
    with self.assertRaises(NameError):  # axis_name will be undefined
      utils.tree_psum(jnp.array([1]), axis_name="i")

  def testAxisNameMismatch(self):
    data = jnp.array([1])
    with self.assertRaises(NameError):
      jax.pmap(lambda x: utils.tree_psum(x, axis_name="i"), axis_name="j")(data)


class MakeAsyncTest(absltest.TestCase):

  def testBaseCase(self):
    """Tests correct execution for single call."""
    r = []
    async_fn = utils.make_async()(lambda: r.append("a"))
    async_fn()
    time.sleep(1)
    self.assertListEqual(r, ["a"])

  def testNonBlocking(self):
    """Tests async function doesn't block the main thread."""
    r = []
    async_fn = utils.make_async()(lambda: r.append((time.sleep(5), "a")))
    r.append((None, "b"))
    async_fn().result()
    self.assertListEqual(r, [(None, "b"), (None, "a")])

  def testSerialExecution(self):
    """Tests multiple calls to async function execute serially."""
    r = []
    a = lambda: r.append((time.sleep(5), "a"))
    b = lambda: r.append((None, "b"))
    async_fn = utils.make_async()(lambda f: f())
    async_fn(a)
    async_fn(b).result()
    self.assertListEqual(r, [(None, "a"), (None, "b")])

  def testErrorOnNextCall(self):
    """Tests background thread error raised in main thread on next call."""

    @utils.make_async()
    def async_fn():
      raise ValueError()

    # First call will trigger an error in the background thread.
    async_fn()

    with self.assertRaises(ValueError):
      # Background thread error will be raised in the main thread on next call
      async_fn()

  def testSubsequentCallsDontRun(self):
    """Tests that subsequent calls don't run after an error has occurred."""

    runs = []

    @utils.make_async()
    def async_fn():
      runs.append(None)
      raise ValueError()

    # First call will trigger an error in the background thread.
    async_fn()

    for _ in range(2):
      with self.assertRaises(ValueError):
        # Background thread error will be raised in the main thread on
        # subsequent calls and _bad_function will not be run.
        async_fn()

    self.assertListEqual(runs, [None])

  def testErrorInBackgroundThread(self):
    """Tests background thread raises the error."""

    @utils.make_async()
    def async_fn():
      raise ValueError()

    future = async_fn()  # pylint: disable=assignment-from-no-return
    self.assertIsNotNone(future.exception())


class TestBroadcast(absltest.TestCase):

  def test_bcast_local_devices(self):
    self.assertEqual(utils.bcast_local_devices(jnp.zeros([])),
                     jnp.zeros([jax.local_device_count()]))

    self.assertEqual(utils.bcast_local_devices(jnp.ones([])),
                     jnp.ones([jax.local_device_count()]))

  def test_bcast_local_devices_empty_tree(self):
    self.assertIsNone(utils.bcast_local_devices(None))
    self.assertEqual(utils.bcast_local_devices({}), {})

  def test_bcast_local_devices_tree(self):
    num_devices = jax.local_device_count()
    tree = utils.bcast_local_devices({"ones": jnp.ones([]),
                                      "zeros": jnp.zeros([])})
    self.assertEqual(tree, {"ones": jnp.ones([num_devices]),
                            "zeros": jnp.zeros([num_devices])})


class TestLogActivity(absltest.TestCase):

  @mock.patch("jaxline.utils.logging.info")
  def test_log_success(self, mock_info):
    """Tests that logging an activity is successful."""

    with utils.log_activity("for test"):
      pass

    mock_info.assert_any_call("[jaxline] %s starting...", "for test")
    mock_info.assert_any_call("[jaxline] %s finished.", "for test")

  @mock.patch("absl.logging.exception")
  @mock.patch("absl.logging.info")
  def test_log_failure(self, mock_info, mock_exc):
    """Tests that an error thrown by an activity is correctly caught."""

    with self.assertRaisesRegex(ValueError, "Intentional"):
      with utils.log_activity("for test"):
        raise ValueError("Intentional")

    mock_info.assert_any_call("[jaxline] %s starting...", "for test")
    mock_exc.assert_any_call("[jaxline] %s failed with error.", "for test")


class TestSpecializeRngHostDevice(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestSpecializeRngHostDevice, cls).setUpClass()
    rng = jax.random.PRNGKey(0)
    cls.rng = jnp.broadcast_to(
        rng, (jax.local_device_count(),) + rng.shape)

  def test_unique_device(self):
    """Tests that rngs are unique across devices."""

    mode = "unique_host_unique_device"
    host_id_devices = utils.host_id_devices_for_rng(mode)
    specialize_func = jax.pmap(functools.partial(
        utils.specialize_rng_host_device, axis_name="i",
        mode=mode), axis_name="i")

    rng = specialize_func(self.rng, host_id_devices)

    self.assertEqual(_num_unique_keys(rng), jax.local_device_count())

  def test_same_device(self):
    """Tests rngs are same across devices."""

    mode = "unique_host_same_device"
    host_id_devices = utils.host_id_devices_for_rng(mode)
    specialize_func = jax.pmap(functools.partial(
        utils.specialize_rng_host_device, axis_name="i",
        mode=mode), axis_name="i")
    rng = specialize_func(self.rng, host_id_devices)

    self.assertEqual(_num_unique_keys(rng), 1)

  def test_unique_host(self):
    """Tests rngs unique between hosts."""

    mode = "unique_host_same_device"
    with mock.patch.object(utils.jax, "process_index", return_value=0):
      host_id_devices = utils.host_id_devices_for_rng(mode)
      specialize_func = jax.pmap(functools.partial(
          utils.specialize_rng_host_device, axis_name="i",
          mode=mode), axis_name="i")
      rng0 = specialize_func(self.rng, host_id_devices)
    with mock.patch.object(utils.jax, "process_index", return_value=1):
      host_id_devices = utils.host_id_devices_for_rng(mode)
      specialize_func = jax.pmap(functools.partial(
          utils.specialize_rng_host_device, axis_name="i",
          mode=mode), axis_name="i")
      rng1 = specialize_func(self.rng, host_id_devices)

    keys = jnp.concatenate([rng0, rng1], axis=0)
    self.assertEqual(_num_unique_keys(keys), 2)


class TestRendezvous(absltest.TestCase):

  def test_rendezvous(self):
    """Test that rendezvous doesn't fail."""

    utils.rendezvous()


class TestJaxlineDisablePmapJit(absltest.TestCase):

  @mock.patch.object(utils.chex, "fake_pmap_and_jit", autospec=True)
  def test_pmap_jit_disabled(self, mock_fake_pmap_and_jit):
    """Tests pmap/jit are disabled if --jaxline_disable_pmap_jit is set."""

    with self.subTest("PmapJitNotDisabled"):
      with flagsaver.flagsaver(jaxline_disable_pmap_jit=False):
        utils.disable_pmap_jit(lambda: None)()
        mock_fake_pmap_and_jit.assert_not_called()

    with self.subTest("PmapJitDisabled"):
      with flagsaver.flagsaver(jaxline_disable_pmap_jit=True):
        utils.disable_pmap_jit(lambda: None)()
        mock_fake_pmap_and_jit.assert_called_once()


class DoubleBufferTest(absltest.TestCase):

  def test_double_buffer(self):
    if jax.default_backend() != "gpu":
      self.skipTest("Only necessary on GPU.")

    n = jax.local_device_count()
    dataset = it.repeat(np.ones([n]))
    iterator = iter(utils.double_buffer(dataset))

    batch_ptrs = []
    while len(batch_ptrs) < 4:
      batch = next(iterator)
      ptrs = [b.unsafe_buffer_pointer() for b in batch.device_buffers]
      batch_ptrs.append(ptrs)
      del batch

    self.assertEqual(batch_ptrs[0], batch_ptrs[2])
    self.assertEqual(batch_ptrs[1], batch_ptrs[3])
    self.assertNotEqual(batch_ptrs[0], batch_ptrs[1])
    self.assertNotEqual(batch_ptrs[2], batch_ptrs[3])


def _run_periodic_growth_logger(
    interval, logging_growth_ratios, start_step, end_step
):
  data = []
  logger = utils.PeriodicAction(
      fn=lambda step, _: data.append(step),  # fn logs step to data
      interval_type="steps",
      interval=interval,
      logging_growth_ratios=logging_growth_ratios,
  )

  for step in range(start_step, end_step):
    logger(time.time(), step+1, {"fake_data": 0})  # pytype: disable=wrong-arg-types  # jax-ndarray
  logger.wait_to_finish()
  return data


class PeriodicActionTest(absltest.TestCase):
  """Tests for PeriodicAction."""

  def test_log_growth_ratios(self):
    """Checks a specific value of growth ratios logs as expected."""
    data = _run_periodic_growth_logger(
        interval=1_000_000,  # Large interval that won't get called early on.
        logging_growth_ratios=[1, 2, 5, 10],
        start_step=0,
        end_step=1010,
    )
    target_data = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    self.assertEqual(data, target_data)

  def test_log_growth_ratios_small(self):
    """Checks a specific value of growth ratios logs as expected."""
    data = _run_periodic_growth_logger(
        interval=1_000_000,  # Large interval that won't get called early on.
        logging_growth_ratios=[1, 2, 5],
        start_step=0,
        end_step=1010,
    )
    # Should still include powers of 10 even if not in the
    # logging_growth_ratios!
    target_data = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    self.assertEqual(data, target_data)

  def test_log_growth_ratios_not_triggered_at_large_steps(self):
    data = _run_periodic_growth_logger(
        interval=5000,
        logging_growth_ratios=[1, 2, 3, 4, 5],
        start_step=399_900,
        end_step=400_100,
    )
    # Should only include steps due to the regular interval.
    target_data = [400_000]
    self.assertEqual(data, target_data)

  def test_log_last_step(self):
    """Checks that when it's enabled, we also log the last step."""
    data = []
    logger = utils.PeriodicAction(
        fn=lambda step, _: data.append(step),  # fn logs step to data
        interval_type="steps",
        interval=200,
        end_step_to_action=1010,
    )

    for step in range(1010):
      logger(time.time(), step+1, {"fake_data": 0})  # pytype: disable=wrong-arg-types  # jax-ndarray
    logger.wait_to_finish()

    # Check that we got the results that we expected
    target_data = [200, 400, 600, 800, 1000, 1010]
    self.assertEqual(data, target_data)


class DummyExperiment(experiment.AbstractExperiment):
  """An experiment for testing the checkpointer."""

  CHECKPOINT_ATTRS = {
      "_params": "params",
  }
  NON_BROADCAST_CHECKPOINT_ATTRS = {
      "_non_bcast_params": "non_bcast_params",
  }

  def __init__(self, mode: str):
    super().__init__(mode=mode)
    self._params = {"a": jnp.array([0])}
    self._non_bcast_params = {"b": jnp.array(1), "c": 2, "d": None}

  def step(self, **kwargs):
    """Only needed for API matching."""

  def evaluate(self, *args, **kwargs):
    """Only need for API matching."""


def _get_checkpoint_history():
  return [utils.SnapshotNT(s.id, s.pickle_nest.to_dict())
          for s in utils.GLOBAL_CHECKPOINT_DICT["ckpt_series"].history]


class TestInMemoryCheckpointer(absltest.TestCase):

  def setUp(self):
    super().setUp()
    utils.GLOBAL_CHECKPOINT_DICT = {}
    config = config_dict.ConfigDict(
        {"max_checkpoints_to_keep": 5})
    self._checkpointer = utils.InMemoryCheckpointer(config, "train")
    self._state = self._checkpointer.get_experiment_state("ckpt_series")
    self._state.global_step = 0
    self._state.experiment_module = DummyExperiment("train")

  def test_save_base_case(self):
    """Tests correct behavior for initial save."""

    self._checkpointer.save("ckpt_series")

    self.assertListEqual(
        _get_checkpoint_history(), [
            utils.SnapshotNT(0, {
                "global_step": 0,
                "experiment_module": {
                    "params": {
                        "a": jnp.array(0)},
                    "non_bcast_params": {
                        "b": jnp.array(1),
                        "c": 2,
                        "d": None}}}),
        ])

  def test_save_with_runahead(self):
    """Tests updating current state doesn't change previous snapshots."""

    self._checkpointer.save("ckpt_series")
    self._state.experiment_module._params["a"] = jnp.array([1])
    self._state.experiment_module._non_bcast_params["b"] = jnp.array(2)
    self._state.experiment_module._non_bcast_params["c"] = 3
    self._state.global_step += 1
    self._checkpointer.save("ckpt_series")

    self.assertListEqual(
        _get_checkpoint_history(), [
            utils.SnapshotNT(0, {
                "global_step": 0,
                "experiment_module": {
                    "params": {
                        "a": jnp.array(0)},
                    "non_bcast_params": {
                        "b": jnp.array(1),
                        "c": 2,
                        "d": None}}}),
            utils.SnapshotNT(1, {
                "global_step": 1,
                "experiment_module": {
                    "params": {
                        "a": jnp.array(1)},
                    "non_bcast_params": {
                        "b": jnp.array(2),
                        "c": 3,
                        "d": None}}}),
        ])

  def test_restore(self):
    """Checks that checkpointable state is correctly restored."""

    for i in range(6):
      self._state.experiment_module._params["a"] = jnp.array([i])
      self._state.experiment_module._non_bcast_params["b"] = jnp.array(i+1)
      self._state.global_step += 1
      self._checkpointer.save("ckpt_series")

    self._checkpointer.restore("ckpt_series")
    current_state = self._checkpointer.get_experiment_state("ckpt_series")
    self.assertEqual(current_state.global_step, 6)
    self.assertEqual(
        current_state.experiment_module._params["a"], jnp.array([5]))
    self.assertEqual(
        current_state.experiment_module._non_bcast_params["b"], jnp.array(6))
    self.assertEqual(
        current_state.experiment_module._non_bcast_params["c"], 2)
    self.assertIsNone(
        current_state.experiment_module._non_bcast_params["d"])

  def test_max_checkpoints_to_keep(self):
    """Checks that max_checkpoints_to_keep is respected."""

    for i in range(10):
      self._state.experiment_module._params["a"] = jnp.array([i])
      self._state.experiment_module._non_bcast_params["b"] = jnp.array(i+1)
      self._state.global_step += 1
      self._checkpointer.save("ckpt_series")

    checkpoint_history = _get_checkpoint_history()
    self.assertLen(checkpoint_history, 5)
    self.assertListEqual(
        [checkpoint_history[0], checkpoint_history[-1]], [
            utils.SnapshotNT(5, {
                "global_step": 6,
                "experiment_module": {
                    "params": {
                        "a": jnp.array(5)},
                    "non_bcast_params": {
                        "b": jnp.array(6),
                        "c": 2,
                        "d": None}}}),
            utils.SnapshotNT(9, {
                "global_step": 10,
                "experiment_module": {
                    "params": {
                        "a": jnp.array(9)},
                    "non_bcast_params": {
                        "b": jnp.array(10),
                        "c": 2,
                        "d": None}}}),
        ])

if __name__ == "__main__":
  absltest.main()
