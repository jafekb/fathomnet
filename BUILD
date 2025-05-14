load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "train_model",
    srcs = ["train_model.py"],
    deps = [
        ":create_subset",
        ":fathomnet_dataset",
        ":classifier",
        ":preprocessing",
        "//base/py",
        "//deps/py:torch",
        requirement("pandas"),
        requirement("scikit-learn"),
    ],
)

py_binary(
    name = "bioclip",
    srcs = ["bioclip.py"],
    deps = [
        ":fathomnet_dataset",
        "//deps/py:torch",
        requirement("pandas"),
        requirement("matplotlib"),
        requirement("scikit-learn"),
        requirement("open-clip-torch"),
    ],
)

py_binary(
    name = "eval_model",
    srcs = ["eval_model.py"],
    deps = [
        ":fathomnet_dataset",
        ":classifier",
        ":preprocessing",
        "//base/py",
        "//deps/py:torch",
        requirement("pandas"),
        requirement("scikit-learn"),
    ],
)

py_library(
    name = "create_subset",
    srcs = ["create_subset.py"],
    deps = [
    ],
)

py_library(
    name = "fathomnet_dataset",
    srcs = ["fathomnet_dataset.py"],
    deps = [
        requirement("pandas"),
        requirement("scikit-learn"),
        "//deps/py:torch",
    ],
)

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
    deps = [
        requirement("pandas"),
        requirement("scikit-learn"),
        "//deps/py:torch",
    ],
)

py_library(
    name = "preprocessing",
    srcs = ["preprocessing.py"],
    deps = [
        "//deps/py:torch",
    ],
)
