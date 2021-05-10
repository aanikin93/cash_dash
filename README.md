# CashDash
A piece of glory in investing stats collection through Tinkoff Investment API


## Prerequisites

Note: currently tested on MacOS Catalina, python 3.9

Ensure that you have installed the following libs (usually go within [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)(https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installation)

- numpy
- pandas
- plotly.dash

```bash
# install plotly.dash
pip install dash

# install tinkoff openapi, refer to https://github.com/Awethon/open-api-python-client
pip install -i https://test.pypi.org/simple/ --extra-index-url=https://pypi.org/simple/ tinkoff-invest-openapi-client
```

## Running dashboard

In order to run dash board you'll need to generate token for authentification in Tinkoff OpenAPI:
 - [How to generate token](https://tinkoffcreditsystems.github.io/invest-openapi/auth/)
 
Then do

```bash
# you may specify the start date from which to collect info in form of YYYY-MM-DD
python run_dash.py --token "put your awesome token here" --start_date="2020-01-01"
```

This will generate a web-app running locally on your machine with the following link (by default):
```bash
Dash is running on http://127.0.0.1:8050/
```