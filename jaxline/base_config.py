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
"""Base config."""

from ml_collections import config_dict


def validate_keys(base_cfg, config, base_filename="base_config.py"):
  """Validates that the config "inherits" from a base config.

  Args:
    base_cfg (`ConfigDict`): base config object containing the required fields
      for each experiment config.
    config (`ConfigDict`): experiment config to be checked against base_cfg.
    base_filename (str): file used to generate base_cfg.

  Raises:
    ValueError: if base_cfg contains keys that are not present in config.
  """

  for key in base_cfg.keys():
    if key not in config:
      raise ValueError("Key {!r} missing from config. This config is required "
                       "to have keys: {}. See {} for details.".format(
                           key, list(base_cfg.keys()), base_filename))
    if (isinstance(base_cfg[key], config_dict.ConfigDict) and
        config[key] is not None):
      validate_keys(base_cfg[key], config[key])


def check_constraints(config):
  """Validates that the config parameters comply to specific set of rules.

  Args:
    config (`ConfigDict`): experiment config to be checked against base_cfg.

  Raises:
    ValueError: if config has train_checkpoint_all_hosts set to True and the
      resulting interval type for checkpointing is time-based (secs).
  """

  if config.train_checkpoint_all_hosts:
    if config.checkpoint_interval_type not in ["steps", None] or (
        config.checkpoint_interval_type is None and
        config.interval_type != "steps"):
      raise ValueError(
          "Invalid interval type selected for the experiment. "
          'When "train_checkpoint_all_hosts = True" True, you need to specify '
          '"checkpoint_interval_type = steps". This is to avoid saving '
          "checkpoints with different global_step on different hosts, which "
          "causes hosts to run out of sync upon restore. Got: "
          f"train_checkpoint_all_hosts: {config.train_checkpoint_all_hosts}, "
          f"interval_type: {config.interval_type}, "
          f"checkpoint_interval_type: {config.checkpoint_interval_type}.")

    if (config.periodic_action_growth_ratios is not None
        and config.interval_type != "steps"):
      raise ValueError(
          f"Invalid interval type {config.interval_type}."
          "When you set periodic_action_growth_ratios you must use "
          "config.interval_type='steps'."
      )


def validate_config(config):
  validate_keys(get_base_config(), config)
  check_constraints(config)


def get_base_config():
  """Returns base config object for an experiment."""
  config = config_dict.ConfigDict()
  config.experiment_kwargs = config_dict.ConfigDict()

  config.training_steps = 10000  # Number of training steps.

  config.interval_type = "secs"
  config.save_checkpoint_interval = 300
  config.log_tensors_interval = 60

  # This is an optional config to allow for logging at growth ratios.
  # This logging is *in addition* to the standard linear intervals.
  # For example, setting [1, 2, 5, 10] will log at [1, 2, 5, 10, 20, 50, ...]
  # Should only be used with interval_type="steps".
  config.periodic_action_growth_ratios = None
  config.log_train_data_interval = 120.0  # None to turn off
  config.log_async = True
  # Log from all training hosts. You can facet on jax_process_index.
  config.log_all_hosts = False

  # Overrides of `interval_type` for specific periodic operations. If `None`,
  # we use the value of `interval_type`.
  config.logging_interval_type = None
  config.checkpoint_interval_type = None

  # If set to True we save the initial checkpoint before executing the first
  # train step. This speeds up eval worker start time as they can immediately
  # begin compilation, and sets a reference point for init model performance.
  config.save_initial_train_checkpoint = False

  # If set to True we checkpoint on all hosts, which may be useful
  # for model parallelism. Otherwise we checkpoint on host 0.
  config.train_checkpoint_all_hosts = False
  config.best_checkpoint_all_hosts = False

  # If True, asynchronously logs training data from every training step.
  config.log_all_train_data = False

  # If true, run evaluate() on the experiment once before you load a checkpoint.
  # This is useful for getting initial values of metrics at random weights, or
  # when debugging locally if you do not have any train job running.
  config.eval_initial_weights = False

  # When True, the eval job immediately loads a checkpoint runs evaluate()
  # once, then terminates.
  config.one_off_evaluate = False

  # Number of checkpoints to keep by default
  config.max_checkpoints_to_keep = 5

  # Settings for the RNGs used during training and evaluation.
  config.random_seed = 42
  config.random_mode_train = "unique_host_unique_device"
  config.random_mode_eval = "same_host_same_device"

  # If True, use the legacy behavior for random seeds where
  # PRNGKey(config.random_seed) is used for both initialization of the
  # experiment class and initialization of the training step rng
  # (without any splitting beforehand).
  config.legacy_random_seed_behavior = False

  # The metric (returned by the step function) used as a fitness score.
  # It saves a separate series of checkpoints corresponding to
  # those which produce a better fitness score than previously seen.
  # By default it is assumed that higher is better, but this behaviour can be
  # changed to lower is better, i.e. behaving as a loss score, by setting
  # `best_model_eval_metric_higher_is_better = False`.
  # If `best_model_eval_metric` is empty (the default), best checkpointing is
  # disabled.
  config.best_model_eval_metric = ""
  config.best_model_eval_metric_higher_is_better = True

  return config
