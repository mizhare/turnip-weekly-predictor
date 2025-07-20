from flask import Flask, render_template, request
import sys
from pathlib import Path

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

        num_parciais = len([p for p in partial_prices if p is not None])

        full_prices = partial_prices[:num_parciais] + predicted_prices
        while len(full_prices) < 12:
            full_prices.append(None)

        labels = [
            "Mon AM", "Mon PM", "Tue AM", "Tue PM",
            "Wed AM", "Wed PM", "Thu AM", "Thu PM",
            "Fri AM", "Fri PM", "Sat AM", "Sat PM"
        ]

        resultado_texto = "<br>".join(
            f"{label}: {'❓' if price is None else f'{price:.2f}'}"
            for label, price in zip(labels, full_prices)
        )

        probs_text = "<br>".join(
            f"{pattern}: {prob*100:.2f}%"
            for pattern, prob in pattern_probabilities.items()
        )

        predicted_start = num_parciais
        ci_labels = labels[predicted_start:]

        ci_text = "<br>".join(
            f"{label}: {mean:.2f} ± {std:.2f}"
            for label, (mean, std) in zip(ci_labels, confidence_intervals)
        )

        resultado_texto += f"<br><br>🔮 Predicted pattern: <b>{predicted_pattern}</b><br><br>"
        resultado_texto += probs_text + "<br><br>"
        resultado_texto += "<b>Predicted prices with confidence intervals:</b><br>" + ci_text

        return render_template("index.html", prediction=resultado_texto)

    except Exception as e:
        return render_template("index.html", prediction=f"Erro: {str(e)}")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))