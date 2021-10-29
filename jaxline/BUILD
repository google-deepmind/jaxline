load("//learning/deepmind/jax:build_defs.bzl", "jax_py_test")

# Description: Jaxline is a user-friendly framework for ML research in JAX.
package(default_visibility = [
    ":friends",
])

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "friends",
    includes = ["//learning/deepmind:visibility"],
    packages = [
        # dm_friend
        # dm_research
        # ithaca/...
    ],
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
        # future
        # jax
        # ml_collections/config_dict
        # typing_extensions
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
        # proto2_pure_python  # Automatically added go/proto_python_default
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
        # ml_collections/config_dict
        # ml_collections/config_flags
        # numpy
        # tensorflow
    ],
)

jax_py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tpu = False,  # Tests fail assuming single device, 1x1 has 2.
    deps = [
        ":utils",
        # proto2_pure_python  # Automatically added go/proto_python_default
        # absl/testing:absltest
        # absl/testing:flagsaver
        # jax
        # numpy
    ],
)
