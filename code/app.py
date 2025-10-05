from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash
import numpy as np
from model_pipeline import predict_car_price, fill_missing_values
import matplotlib.pyplot as pl

# Initialize Dash app
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    suppress_callback_exceptions=True
)
app.title = "Car Price Predictor"

# Layout with URL routing
app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# Main Page Layout
main_page = dbc.Container([
    html.H1("Welcome to Car Classifer", className="text-center my-5"),

    html.P(
        "This web application classify a car in to class from Class 0, Class 1, Class 2 and Class 3 based on a few key features. These classes represent different price ranges, helping users understand the market value of their vehicles.",
        "Simply provide the required inputs on the classification Page and get an instant result!",
        className="lead text-center mb-4"
    ),

    dbc.Card([
        dbc.CardHeader(html.H4("How Prediction Works")),
        dbc.CardBody([
            html.P(
                "The app uses a trained machine learning model to classify a car according to provided input details. "
                "The model has learned from historical car data including features like transmission type and engine power."
            ),
            html.Ul([
                html.Li([
                        html.B("Transmission Type: "),  # note the space
                        "Manual (0) or Automatic (1)."
                ]),
                html.Li([
                        html.B("Max Power: "),
                        "Maximum engine power in bhp. Higher power often increases the price."
                ])
            ]),
            html.P(
                "After entering these details on the Prediction Page, "
                "the model outputs the class which indicates the car's price range."
            ),
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(
            dbc.Button("Go to Prediction Page", href="/predict", color="primary", className="d-block mx-auto"),
            width=12
        )
    ])
], className="my-4")

# Prediction Page Layout
prediction_page = dbc.Container([
    html.H1("Car Price Prediction", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Transmission Type"),
            dcc.Dropdown(
                id="transmission",
                options=[
                    {"label": "Manual", "value": 0},
                    {"label": "Automatic", "value": 1}
                ],
                placeholder="Select Transmission Type",
                className="mb-3"
            ),

            html.Label("Max Power (bhp)"),
            dcc.Input(
                id="max_power", type="number",
                placeholder="Enter Max Power",
                className="form-control mb-3"
            ),

            dbc.Button("Predict", id="submit-btn", color="primary", className="mt-3"),
            html.Br(),
            html.Br(),
            dbc.Button("Try Again", id="try-again-btn", color="warning", className="me-2"),
            dbc.Button("Back to Home", href="/", color="secondary")
        ], md=6)
    ]),

    html.Hr(),
    html.Div(id="prediction-output", className="h4 text-success text-center mt-3")
])

# Routing Callback
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/predict":
        return prediction_page
    return main_page

# Prediction Callback
@app.callback(
    Output("transmission", "value"),
    Output("max_power", "value"),
    Output("prediction-output", "children"),
    Input("submit-btn", "n_clicks"),
    Input("try-again-btn", "n_clicks"),
    State("transmission", "value"),
    State("max_power", "value"),
    prevent_initial_call=True
)

# app.py
def handle_prediction_or_reset(submit_clicks, try_again_clicks, transmission, max_power):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    return predict_or_reset_logic(button_id, transmission, max_power)

def predict_or_reset_logic(button_id, transmission, max_power):
    if button_id == "submit-btn":
        transmission, max_power = fill_missing_values(transmission, max_power)
        sample = np.array([[1, transmission, max_power]])
        prediction = predict_car_price(sample)
        return transmission, max_power, f"Classified Car Category: {prediction}"
    elif button_id == "try-again-btn":
        return None, None, ""





# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8050)
