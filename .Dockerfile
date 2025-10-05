FROM python:3.11.4-bookworm

WORKDIR /root/code

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install pytest
RUN pip3 install pytest-mock
RUN pip3 install joblib
RUN pip3 install plotly
RUN pip3 install matplotlib
RUN pip3 install mlflow


# Testing module
RUN pip3 install dash[testing]

COPY ./code /root/code

CMD tail -f /dev/null
