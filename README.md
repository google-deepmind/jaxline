# JAXline - Experiment framework for JAX

## What is JAXline

JAXline is a distributed JAX training and evaluation framework.
It is designed to be forked, covering only the most general aspects of
experiment boilerplate. This ensures that it can serve as an effective starting
point for a wide variety of use cases.

Many users will only need to fork the
[`experiment.py`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
file and rely on JAXline for everything else. Other users with more custom
requirements will want to (and are encouraged to) fork other components of
JAXline too, depending on their particular use case.

### Contents

*   [Quickstart](#quickstart)
*   [Checkpointing](#checkpointing)
*   [Logging](#logging)
*   [Launching](#launching)
*   [Distribution strategy](#distribution-strategy)
*   [Random number handling](#random-number-handling)
*   [Debugging](#debugging)
*   [Contributing](#contributing)

## Quickstart

### Installation

JAXline is written in pure Python, but depends on C++ code via JAX and
TensorFlow (the latter is used for writing summaries).

Because JAX / TensorFlow installation is different depending on your CUDA
version, JAXline does not list JAX or TensorFlow as a dependencies in
`requirements.txt`.

First, follow the instructions to install
[JAX](https://github.com/google/jax#installation) and
[TensorFlow](https://github.com/tensorflow/tensorflow#install)
respectively with the relevant accelerator support.

Then, install JAXline using pip:

```bash
$ pip install git+https://github.com/deepmind/jaxline
```

### Building your own experiment

1.  Create an `experiment.py` file and inside it define an `Experiment` class
    that inherits from
    [`experiment.AbstractExperiment`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py).
2.  Implement the methods required by
    `AbstractExperiment` in your own `Experiment` class (i.e. the
    `abstractmethod`s). Optionally override the default implementations of `AbstractExperiment`'s other methods.
3.  Define a `config`, either in `experiment.py` or elsewhere, defining any
    settings that you do not wish to inherit from
    [`base_config`](https://github.com/deepmind/jaxline/tree/master/jaxline/base_config.py).
    At the very least this will include `config.experiment_kwargs` to define the
    config required by your `Experiment`. Make sure this `config` object is
    included in the `flags` accessible to `experiment.py`.
4.  Add the following lines to the bottom of your `experiment.py` to ensure that
    your `Experiment` object is correctly passed through to
    [`platform.py`](https://github.com/deepmind/jaxline/tree/master/jaxline/platform.py):

    ```
    if __name__ == '__main__':
      flags.mark_flag_as_required('config')
      platform.main(Experiment, sys.argv[1:])
    ```

4.  Run your `experiment.py`.

## Checkpointing

So far this version of JAXline only supports in-memory checkpointing, as handled
by our
[`InMemoryCheckpointer`](https://github.com/deepmind/jaxline/tree/master/jaxline/utils.py)
It allows you to save in memory multiple separate checkpoint series in your
train and eval jobs (see below).

The user is expected to override the
[`CHECKPOINT_ATTRS`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
and
[`NON_BROADCAST_CHECKPOINT_ATTRS`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
dicts (at least one of these) in order to map checkpointable attributes of their
own `Experiment` class to names they wish them to be stored under in the
checkpoint.
`CHECKPOINT_ATTRS` specifies jax `DeviceArrays` for which jaxline should only
take the first slice (corresponding to device 0) for checkpointing.
`NON_BROADCAST_CHECKPOINT_ATTRS` specifies any other picklable object that
jaxline should checkpoint whole.

You can specify the frequency with which to save checkpoints, as well as whether
to checkpoint based on step or seconds, by setting the
`save_checkpoint_interval` and `interval_type`  config flags
[here](https://github.com/deepmind/jaxline/tree/master/jaxline/base_config.py).

`config.max_checkpoints_to_keep` can be used to specify the maximum number of
checkpoints to keep. By default this is set to 5.

By setting `config.best_model_eval_metric`, you can specify which value in the
`scalars` dictionary returned by your
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
function to use as a 'fitness score'. JAXline will then save a separate series
of checkpoints corresponding to steps at which the fitness score is better than
previously seen. Depending on whether you are maximizing or minimizing the eval
metric, set `config.best_model_eval_metric_higher_is_better` to True or False.

## Logging

So far this version of JAXline only supports logging to Tensorboard via our
[`TensorBoardLogger`](https://github.com/deepmind/jaxline/tree/master/jaxline/platform.py)

The user is expected to return a dictionary of scalars from their
[`step`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
and
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
methods, and
[`TensorBoardLogger.write_scalars`](https://github.com/deepmind/jaxline/tree/master/jaxline/platform.py)
will periodically write these scalars to `TensorBoard`.

All logging will happen asynchronously to the main thread so as not to interrupt
the training loop.

You can specify the frequency with which to log, as well as whether to log by
step or by seconds, by setting the `log_train_data_interval` and `interval_type`
config flags [here](https://github.com/deepmind/jaxline/tree/master/jaxline/base_config.py).
If `config.log_all_train_data` is set to `True` (`False` by default) JAXline
will cache the scalars from intermediate steps and log them all at once at the
end of the period.

JAXline passes the
[`TensorBoardLogger`](https://github.com/deepmind/jaxline/tree/master/jaxline/platform.py)
instance through to the
[`step`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
and
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
methods to allow the user to perform additional logging inside their
`Experiment` class if they so wish. A particular use case for this is if you
want to write images, which can be achieved via
[`ExperimentWriter.write_images`](https://github.com/deepmind/jaxline/tree/master/jaxline/platform.py).


## Launching

So far this version of JAXline does not support launching remotely.

## Distribution strategy

JAX makes it super simple to distribute your jobs across multiple hosts and
cores. As such, JAXline leaves it up to the user to implement distributed
training and evaluation.

Essentially, by decorating a function with
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
you tell JAX to slice the inputs along the first dimension and then run the
function in parallel on each input slice, across all available local devices (or
a subset thereof). In other words,
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
invokes the single-program multiple-data (SPMD) paradigm. Then by using
[`jax.lax`](https://jax.readthedocs.io/en/latest/jax.lax.html) collective
communications operations from within your pmapped function, you can tell JAX to
communicate results between all devices _on all hosts_. For example, you may
want to use [`jax.lax.psum`](https://jax.readthedocs.io/en/latest/jax.lax.html)
to sum up the gradients across all devices on all hosts, and return the result
to each device (an all-reduce).

JAX will then automatically detect which devices are available on each host
allowing
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
and [`jax.lax`](https://jax.readthedocs.io/en/latest/jax.lax.html) to work their
magic.

One very important thing to bear in mind is that each time you call
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap),
a separate TPU program will be compiled for the computation it wraps. Therefore
you do not want to be doing this regularly! In particular, for a standard ML
experiment you will want to call
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
once to wrap your parameter update function,
and then you call this wrapped function on each step, rather than calling
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
on each step, which will kill your performance! This is a very common mistake
for new JAX starters. Luckily it has quite an extreme downside so should be
easily noticeable. In JAXline we actually call
[`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
once more in
[`next_device_state`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
to wrap our function to update device state between steps, so end up with 2 TPU
programs rather than just 1 (but this adds negligible overhead).

## Random number handling

Random numbers in JAX might seem a bit unfamiliar to users coming from ordinary
`numpy` and `Tensorflow`. In these languages we have global stateful PRNGs.
Every time you call a random op it updates the PRNGs global state. However,
stateful PRNGs in JAX would be incompatible with JAX's functional design
semantics, leading to problems with reproducibility and parallelizability. JAX
introduces stateless PRNGs to avoid these issues. The downside of this is that
the user needs to thread random state through their program, splitting a new
PRNG off from the old one every time they want to draw a new random number. This
can be quite onerous, especially in a distributed setting, where you may have
independent PRNGs on each device.

In JAXline we take care of this for you. On each step, in
[`next_device_state`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py),
we split a new PRNG from the old one, and optionally specialize it to the host
and/or device based on the
`random_mode_train` [config](https://github.com/deepmind/jaxline/tree/master/jaxline/base_config.py)
value you specify. We then pass this new PRNG through to your
[step](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
function to use on that particular step. At evaluation time, we pass a fresh
PRNG to your
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
method, initialized according to the `random_mode_eval`
[config](https://github.com/deepmind/jaxline/tree/master/jaxline/base_config.py) value
you specify. This PRNG will be the same on each call to
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
(as normally you want your evaluation to be deterministic). If you want
different random behaviour on each call, a simple solution would be to fold in
the `global_step` i.e. `jax.random.fold_in(rng, global_step)` at the top of your
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
method.

Of course you are free to completely ignore the PRNGs we pass through to your
[`step`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
and
[`evaluate`](https://github.com/deepmind/jaxline/tree/master/jaxline/experiment.py)
methods and handle random numbers in your own way, should you have different
requirements.

## Debugging

### Post mortem debugging

By setting the flag `--jaxline_post_mortem` (defined
[here](https://github.com/deepmind/jaxline/tree/master/jaxline/utils.py)) on the command-line,
tasks will pause on exceptions (except `SystemExit` and `KeyboardInterrupt`) and
enter post-mortem debugging using pdb. Paused tasks will hang until you attach
a debugger.

### Disabling pmap and jit

By setting the flag `--jaxline_disable_pmap_jit` on the command-line, all pmaps
and jits will be disabled, making it easier to inspect and trace code in a
debugger.

## Citing Jaxline

Please use [this reference](https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt).


## Contributing

Thank you for your interest in JAXline. The primary goal of open-sourcing
JAXline was to allow us to open-source our research more easily. Unfortunately,
we are not currently able to accept pull requests from external contributors,
though we hope to do so in future. Please feel free to open GitHub issues.
