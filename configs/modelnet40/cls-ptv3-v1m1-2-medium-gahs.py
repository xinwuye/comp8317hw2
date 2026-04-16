_base_ = ["./cls-ptv3-v1m1-1-medium-baseline.py"]

model = dict(
    backbone=dict(
        order=("gahs", "gahs-trans"),
        serialization_cfg=dict(
            knn_k=16,
            density_weight=0.5,
            curvature_weight=0.5,
            hilbert_quantile=0.7,
            fine_bits_cap=6,
            eps=1e-6,
        ),
    )
)
