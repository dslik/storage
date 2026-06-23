"""
Regression tests for issue #463:
vdb_benchmark indexing parameter values from config files were not getting used.

Two root causes:
  1. Index tuning parameters nested under an `index_params:` block were
     silently dropped by merge_config_with_args (only one level was walked).
  2. A config value equal to the argparse default was treated as "still
     default" and not applied.

See: https://github.com/mlcommons/storage/issues/463
"""
import argparse

import yaml

from vdbbench.config_loader import merge_config_with_args, _flatten_config


def _make_args():
    args = argparse.Namespace(
        index_type="DISKANN", metric_type="COSINE",
        M=16, ef_construction=200, max_degree=16, search_list_size=200,
        collection_name=None, num_vectors=None, dimension=None,
    )
    args.is_default = {
        "index_type": True, "metric_type": True, "M": True,
        "ef_construction": True, "max_degree": True, "search_list_size": True,
        "collection_name": True, "num_vectors": True, "dimension": True,
    }
    return args


def test_nested_index_params_are_applied():
    cfg = yaml.safe_load("""
    index:
      index_type: HNSW
      metric_type: COSINE
      index_params:
        M: 32
        ef_construction: 100
    """)
    args = merge_config_with_args(cfg, _make_args())
    assert args.index_type == "HNSW"
    assert args.M == 32
    assert args.ef_construction == 100


def test_flat_index_params_still_work():
    cfg = yaml.safe_load("""
    index:
      index_type: HNSW
      M: 64
      ef_construction: 200
    """)
    args = merge_config_with_args(cfg, _make_args())
    assert args.M == 64
    assert args.ef_construction == 200


def test_config_value_equal_to_default_is_applied():
    cfg = yaml.safe_load("""
    index:
      index_type: HNSW
      M: 16
      ef_construction: 100
    """)
    args = merge_config_with_args(cfg, _make_args())
    assert args.M == 16
    assert args.ef_construction == 100


def test_cli_flag_overrides_config():
    cfg = yaml.safe_load("""
    index:
      index_type: HNSW
      index_params:
        M: 32
        ef_construction: 100
    """)
    args = _make_args()
    args.M = 99               # user passed --M 99
    args.is_default["M"] = False
    merge_config_with_args(cfg, args)
    assert args.M == 99
    assert args.ef_construction == 100


def test_diskann_nested_params():
    cfg = yaml.safe_load("""
    index:
      index_type: DISKANN
      index_params:
        max_degree: 96
        search_list_size: 250
    """)
    args = merge_config_with_args(cfg, _make_args())
    assert args.max_degree == 96
    assert args.search_list_size == 250


def test_unknown_config_keys_are_ignored():
    cfg = yaml.safe_load("""
    database:
      max_receive_message_length: 514983574
    index:
      index_type: HNSW
      index_params:
        M: 48
    """)
    args = merge_config_with_args(cfg, _make_args())
    assert args.M == 48


def test_flatten_lifts_nested_leaves():
    flat = _flatten_config(yaml.safe_load("""
    index:
      index_type: HNSW
      index_params:
        M: 32
    """))
    assert flat == {"index_type": "HNSW", "M": 32}


def test_empty_config_is_noop():
    args = _make_args()
    out = merge_config_with_args({}, args)
    assert out.M == 16 and out.index_type == "DISKANN"

