from flask import Flask, render_template, request
import sys
from pathlib import Path
import os
import plotly.graph_objs as go
from plotly.offline import plot

app = Flask(__name__)

model_training_path = Path(__file__).resolve().parent / "model_training"
sys.path.append(str(model_training_path))

from predict import predict_from_partial

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prever", methods=["POST"])
def prever():
    try:
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
        periods = ["am", "pm"]

        partial_prices = []
        for day in days:
            for period in periods:
                key = f"{day}_{period}"
                val_str = request.form.get(key, "").strip()
                if val_str == "":
                    partial_prices.append(None)
                else:
                    partial_prices.append(float(val_str))

        previous_pattern_raw = request.form.get("previous_pattern", "unknown")

        predicted_pattern, predicted_prices, pattern_probabilities, confidence_intervals = predict_from_partial(
            partial_prices=partial_prices,
            model_folder=str(model_training_path),
            previous_pattern_raw=previous_pattern_raw
        )

        num_partial = len([p for p in partial_prices if p is not None])
        full_prices = partial_prices[:num_partial] + predicted_prices
        while len(full_prices) < 12:
            full_prices.append(None)

        labels = [
            "Mon AM", "Mon PM", "Tue AM", "Tue PM",
            "Wed AM", "Wed PM", "Thu AM", "Thu PM",
            "Fri AM", "Fri PM", "Sat AM", "Sat PM"
        ]

        result_text = "<br>".join(
            f"{label}: {'❓' if price is None else f'{price:.2f}'}"
            for label, price in zip(labels, full_prices)
        )

        probs_text = "<br>".join(
            f"{pattern}: {prob*100:.2f}%"
            for pattern, prob in pattern_probabilities.items()
        )

        predicted_start = num_partial
        ci_labels = labels[predicted_start:]

        ci_text = "<br>".join(
            f"{label}: {mean:.2f} ± {std:.2f}"
            for label, (mean, std) in zip(ci_labels, confidence_intervals)
        )

        result_text = f'🔮 Predicted pattern: <span style="color:#e78a4e;">{predicted_pattern}</span><br><br>'

        probs_text = "<br>".join(
            f'{pattern}: <span style="color:#e78a4e;font-family: Patrick Hand, cursive; font-size: 18px;font-weight: bold;">{prob * 100:.2f}%</span>'
            for pattern, prob in pattern_probabilities.items()
        )

        result_text += probs_text + "<br><br>"
        result_text += "<b>Predicted prices with confidence intervals:</b><br>"
        result_text += "<br>".join(
            f'{label}: <span style="color:#e78a4e; font-family: Patrick Hand, cursive; font-size: 18px;font-weight: bold;">{mean:.2f} ± {std:.2f}</span>'
            for label, (mean, std) in zip(ci_labels, confidence_intervals)
        )

        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=[price if price is not None else 0 for price in full_prices],
            marker_color='#FFBCC4'  # cor rosa
        )])
        fig.update_layout(
            title=dict(
                text="<b>Bells</b>",
                x=0.5,
                y=0.83,
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Komika Axis",
                    size=17,
                    color="#5c4033",
                )
            ),
            xaxis=dict(
                title=dict(
                    text="<b>Time of day</b>",
                    font=dict(
                        family="Komika Axis",
                        size=17,
                        color="#5c4033",
                    ),
                    standoff=28
                ),
                showgrid=False,
                zeroline=True,
                zerolinecolor="#5c4033",
            ),
            yaxis=dict(
                range=[0, max([p for p in full_prices if p is not None] + [100]) * 1.2],
                showgrid=True,
                gridcolor="#5c4033",
                zeroline=True,
                zerolinecolor="#5c4033",
            ),
            plot_bgcolor="#f0e5d8",
            paper_bgcolor="#f0e5d8",
            font=dict(
                family='Patrick Hand, cursive',
                size=18,
                color='#5c4033'
            )
        )
        plot_div = plot(fig, output_type='div', include_plotlyjs=True)
        print(plot_div)
        return render_template(
            "index.html",
            prediction=result_text,
            plot_div=plot_div
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))