"""Gradio UI for interactively testing the segmentation model.

Mounted inside the FastAPI app at ``/ui`` via ``gr.mount_gradio_app`` —
shares the same process, port, and model instance. The callback calls the
same :func:`src.app.inference.predict` used by the REST API; no inference
logic is re-implemented here.
"""

from __future__ import annotations

import gradio as gr

from src.app.inference import FEATURE_COLS, predict

# Sensible defaults / ranges from the training data's observed distributions.
# (See the EDA notebook: heavy right-skew, so sliders use log-ish ranges.)
_DEFAULTS: dict[str, float] = {
    "avg_order_value": 100.0,
    "avg_items_per_order": 10.0,
    "num_orders": 4,
    "num_products": 10,
    "total_items": 50,
    "avg_unit_price": 5.0,
    "total_revenue": 1200.0,
    "sum_cancelled_orders": 0,
    "customer_lifetime_days": 200,
    "recency_days": 100,
    "purchase_frequency": 0.5,
}

_MAXS: dict[str, float] = {
    "avg_order_value": 100_000.0,
    "avg_items_per_order": 10_000.0,
    "num_orders": 1_000,
    "num_products": 1_000,
    "total_items": 100_000,
    "avg_unit_price": 10_000.0,
    "total_revenue": 1_000_000.0,
    "sum_cancelled_orders": 1_000,
    "customer_lifetime_days": 1_000,
    "recency_days": 1_000,
    "purchase_frequency": 100.0,
}


def _input_widgets() -> list[gr.components.Component]:
    """Build one input widget per feature, matched to its type."""
    widgets: list[gr.components.Component] = []
    for col in FEATURE_COLS:
        default = _DEFAULTS[col]
        max_value = _MAXS[col]
        if col in {
            "num_orders",
            "num_products",
            "total_items",
            "sum_cancelled_orders",
            "customer_lifetime_days",
            "recency_days",
        }:
            widgets.append(
                gr.Number(
                    value=default,
                    minimum=0,
                    maximum=max_value,
                    precision=0,
                    label=col,
                )
            )
        else:
            widgets.append(
                gr.Number(
                    value=default,
                    minimum=0,
                    maximum=max_value,
                    label=col,
                )
            )
    return widgets


def _predict_callback(*values: float) -> str:
    """Call the shared inference function and format the result for display."""
    features = dict(zip(FEATURE_COLS, values, strict=True))
    result = predict(features)
    return (
        f"**Cluster {result['cluster'] + 1}** — {result['segment']}\n\n"
        f"Assigned segment: **{result['segment']}**"
    )


def build_ui(app: gr.Blocks | None = None) -> None:
    """Build the Gradio Blocks UI and mount it inside the FastAPI app at ``/ui``.



    Args:
        app: The FastAPI app to mount the UI into (passed by main.py).
    """
    with gr.Blocks(title="Customer Segmentation Explorer") as blocks:
        gr.Markdown(
            """
            # Customer Segmentation Explorer

            Enter a customer's behavioral features and the model will assign them
            to one of three segments: **Occasional bulk buyers**,
            **Everyday value shoppers**, or **High-volume resellers**.

            Expected input ranges are based on the training data (log-ish
            scale for the heavily right-skewed features).

            > Disclaimer: predictions are model output, not ground truth.

            """
        )
        with gr.Row():
            with gr.Column():
                inputs = _input_widgets()
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                output = gr.Markdown(label="Prediction")

        predict_btn.click(
            fn=_predict_callback,
            inputs=inputs,
            outputs=output,
        )

    if app is not None:
        gr.mount_gradio_app(app, blocks, path="/ui")
