import argparse
from collections import defaultdict
import pandas as pd
from openapi_client import openapi
import numpy as np

import dash
import dash_table
import dash_html_components as html

from market_watch import MarketInfo
from market_watch import MyPortfolio
from market_watch import DealsHistory


class CashDash:
    def __init__(self, token, start_date="2020-01-01"):
        self._client = openapi.api_client(token)
        self.market_info = MarketInfo(self._client)
        self.portfolio = MyPortfolio(client=self._client, start_date=start_date)
        self.positions = self._get_positions_dict()
        self.ops_df = self._get_all_operations()

    def _get_positions_dict(self):
        return self.portfolio.get_my_positions()

    def _get_all_operations(self):
        return self.portfolio.get_ops_df()

    def _compute_cummulative_deals(self):
        cummulative_deals = defaultdict(list)

        for ticker in set(self.ops_df.ticker):
            print(ticker)
            if ticker is not None and ticker not in ['USD000UTSTOM', 'EUR_RUB__TOM']:
                deals_hist = DealsHistory(ticker=ticker, ops_df=self.ops_df)
                cummulative_deals["currency"].append(self.market_info.get_instruments_info()[ticker]['currency'])
                cummulative_deals["ticker"].append(ticker)
                cummulative_deals["local_non_loss_value"].append(deals_hist.local_non_loss_value)
                cummulative_deals["total_fixed_profit_value"].append(deals_hist.total_profit_value)
                cummulative_deals["opened_position_value"].append(deals_hist.opened_position_value)
                cummulative_deals["commission_paid"].append(deals_hist.commission_paid)
                cummulative_deals["dividends"].append(deals_hist.dividends)
                mpd = self.positions.get(ticker)
                if mpd:
                    print(mpd)
                    non_loss_value, filtered_deals_list = mpd.get_deals_stats()
                    cummulative_deals["global_non_loss_value"].append(non_loss_value)
                    cummulative_deals["num_units_opened"].append(mpd.num_units)
                    cummulative_deals["avg_price"].append(mpd.avg_price)
                else:
                    cummulative_deals["global_non_loss_value"].append(0)
                    cummulative_deals["num_units_opened"].append(0)
                    cummulative_deals["avg_price"].append(0)

        cummulative_df = pd.DataFrame(data=cummulative_deals)
        return cummulative_df

    def run_app(self):
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        cols_list = ["ticker", "opened_position_value", "num_units_opened", "avg_price", "local_non_loss_value",
                     "global_non_loss_value", "commission_paid", "total_fixed_profit_value", "dividends", 'currency']
        cummulative_df = self._compute_cummulative_deals()
        for col in cols_list:
            if col not in ["ticker", "opened_position_value", "currency"]:
                cummulative_df[col] = np.round(cummulative_df[col], 3)
        # cummulative_df["total_fixed_profit_value"] = np.round(cummulative_df["total_fixed_profit_value"], 3)
        # cummulative_df["commission_paid"] = np.round(cummulative_df["commission_paid"], 3)

        app.layout = html.Div([
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[
                    {"name": i, "id": i, "deletable": True, "selectable": True} for i in cols_list
                ],

                style_table={'height': 'auto', 'overflowY': 'auto'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{currency} contains USD',
                            'column_id': 'currency'
                        },
                        'backgroundColor': 'rgb(240, 240, 240)'
                    },
                    {
                        'if': {
                            'filter_query': '{currency} contains RUB',
                            'column_id': 'currency'
                        },
                        'backgroundColor': 'rgb(230, 230, 230)'

                    },
                    {
                        'if': {
                            'filter_query': '{currency} contains EUR',
                            'column_id': 'currency'
                        },
                        'backgroundColor': 'rgb(220, 220, 220)'

                    }
                ],
                style_header={
                    'backgroundColor': 'paleturquoise',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'left',
                    #             'backgroundColor': 'lavender',
                    'minWidth': 95, 'maxWidth': 95, 'width': 95
                },
                data=cummulative_df.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=False,
                selected_columns=[],
                selected_rows=[],
                page_action="native",

            ),
            html.Div(id='datatable-interactivity-container')
        ])

        app.run_server()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running CashDash")
    parser.add_argument("--token", type=str,
                        help="Token for connection to Tinkoff invest API")
    parser.add_argument("--start_date", type=str,
                        help="Date to begin with for downloading stats", default="2020-01-01")

    args = parser.parse_args()

    cd = CashDash(token=args.token, start_date=args.start_date)
    cd.run_app()
