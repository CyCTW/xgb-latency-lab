import argparse
import numpy as np

from .compiler import compile_model


def main():
    p = argparse.ArgumentParser(description="Compile a scalar XGBoost JSON model to native raw-margin inference")
    p.add_argument("model")
    p.add_argument("output_dir")
    p.add_argument("--select-depth", type=int, default=1)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--tree-block-size", type=int, default=0)
    p.add_argument("--predicate-hoist-limit", type=int, default=0)
    p.add_argument("--rank-feature-limit", type=int, default=0,
                   help="Maximum high-use features to encode as exact threshold ranks")
    p.add_argument("--rank-strategy", choices=["binary", "eytzinger", "simd", "bucket", "bucket_split"], default="binary")
    p.add_argument("--rank-bucket-bits", type=int, default=12)
    p.add_argument("--compact-leaf-depth", type=int, default=0)
    p.add_argument("--accumulation-batch", type=int, default=1)
    p.add_argument("--machine-outliner", action="store_true")
    p.add_argument("--traversal-lanes", type=int, default=0, choices=[0,1,2,4,8,12,16,24,32])
    p.add_argument("--traversal-mode", choices=['scalar','vector'], default='scalar')
    p.add_argument("--traversal-load-schedule", choices=['staged','direct','lane'], default='staged')
    p.add_argument("--traversal-prefetch", choices=['none','roots','next','both'], default='none')
    p.add_argument("--traversal-prefetch-distance", type=int, choices=[1,2,4], default=1)
    p.add_argument("--traversal-prefetch-locality", type=int, choices=[0,1,2,3], default=3)
    p.add_argument("--traversal-leaf-layout", choices=['sentinel','self_loop','separate'], default='sentinel')
    p.add_argument("--traversal-leaf-state", choices=['split','packed'], default='split')
    p.add_argument("--traversal-data-layout", choices=['aos','soa','soa8'], default='aos')
    p.add_argument("--traversal-alignment", type=int, choices=[16,64,128,4096], default=16)
    p.add_argument("--hybrid-depth", type=int, default=0,
                   help="Pack subtrees up to this height into shared data traversal (0 disables)")
    p.add_argument("--hybrid-layout", choices=['wide','compact'], default='wide')
    p.add_argument("--hybrid-max-probability", type=float, default=1.0,
                   help="Maximum calibration visit fraction for data subtrees; below 1 requires calibration")
    p.add_argument("--optimization", choices=["O3", "O2", "Os", "Oz"], default="O3")
    p.add_argument("--leaf-table-bits", type=int, default=0,
                   help="Maximum predicates per compile-time leaf table (0 disables, max 8)")
    p.add_argument("--select-policy", choices=["height","profile","cost"], default="height")
    p.add_argument("--select-branch-penalty", type=float, default=4.,
                   help="Cost-policy branch penalty in comparison-equivalent units")
    p.add_argument("--backend", choices=["llvmlite","clang"], default="llvmlite")
    p.add_argument("--calibration", help="Float32-compatible .npy matrix for branch weights")
    args = p.parse_args()
    print(compile_model(args.model, args.output_dir, select_depth=args.select_depth,
                        preload=args.preload,backend=args.backend,
                        tree_block_size=args.tree_block_size,select_policy=args.select_policy,
                        predicate_hoist_limit=args.predicate_hoist_limit,
                        leaf_table_bits=args.leaf_table_bits,
                        select_branch_penalty=args.select_branch_penalty,
                        rank_feature_limit=args.rank_feature_limit,
                        rank_strategy=args.rank_strategy, rank_bucket_bits=args.rank_bucket_bits, compact_leaf_depth=args.compact_leaf_depth,
                        accumulation_batch=args.accumulation_batch, optimization=args.optimization, machine_outliner=args.machine_outliner,
                        hybrid_depth=args.hybrid_depth, hybrid_max_probability=args.hybrid_max_probability,
                        hybrid_layout=args.hybrid_layout,
                        traversal_lanes=args.traversal_lanes, traversal_mode=args.traversal_mode,
                        traversal_leaf_layout=args.traversal_leaf_layout,
                        traversal_leaf_state=args.traversal_leaf_state,
                        traversal_data_layout=args.traversal_data_layout, traversal_alignment=args.traversal_alignment,
                        traversal_load_schedule=args.traversal_load_schedule,
                        traversal_prefetch=args.traversal_prefetch, traversal_prefetch_distance=args.traversal_prefetch_distance,
                        traversal_prefetch_locality=args.traversal_prefetch_locality,
                        calibration=np.load(args.calibration, allow_pickle=False) if args.calibration else None))


if __name__ == "__main__":
    main()
