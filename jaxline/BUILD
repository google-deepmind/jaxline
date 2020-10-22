load("//production/dependency/rpc/testing/hermetic:build_defs.bzl", "py_strict_hermetic_test")

# Description: Jaxline is a user-friendly framework for ML research in JAX.
package(default_visibility = [
    ":friends",
])

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "friends",
    includes = ["//learning/deepmind:visibility"],
)

py_library(
    name = "base_config",
    srcs = ["base_config.py"],
    deps = [
        # ml_collections/config_dict
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        # absl/flags
        # absl/logging
        # chex
        # jax
        # ml_collections/config_dict
        # wrapt
    ],
)

py_test(
    name = "train_test",
    srcs = ["train_test.py"],
    python_version = "PY3",
    deps = [
        ":base_config",
        ":experiment",
        ":train",
        # absl/testing:absltest
        # ml_collections/config_dict
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # absl/flags
        # absl/logging
        # jax
    ],
)

py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # absl/logging
        # jax
        # ml_collections/config_dict
        # numpy
    ],
)

py_library(
    name = "platform",
    srcs = ["platform.py"],
    srcs_version = "PY3",
    deps = [
        ":base_config",
        ":train",
        ":utils",
        # absl/flags
        # absl/logging
        # chex
        # jax
        # ml_collections
        # ml_collections/config_flags
        # numpy
        # tensorflow
    ],
)

py_strict_hermetic_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        # absl/testing:absltest
        # absl/testing:flagsaver
        # jax
        # numpy
    ],
)
