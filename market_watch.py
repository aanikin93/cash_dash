from datetime import datetime, timedelta
from pytz import timezone
import pandas as pd
import numpy as np
import json
from collections import defaultdict


class MarketInfo:
    def __init__(self, client):
        self._client = client
        self._bonds = self._client.market.market_bonds_get().payload.instruments
        self._etfs = self._client.market.market_etfs_get().payload.instruments
        self._stocks = self._client.market.market_stocks_get().payload.instruments
        self._currencies = self._client.market.market_currencies_get().payload.instruments
        self._instruments_info = self.get_instruments_info()

    def get_instruments_info(self):
        instrument_info = dict()
        instruments_list = self._bonds + self._stocks + self._etfs + self._currencies
        for instrument in instruments_list:
            instr = instrument.to_dict()
            instrument_info[instr['ticker']] = instr
        return instrument_info

    def get_instruments_info_by_figi(self):
        instrument_info = dict()
        instruments_list = self._bonds + self._stocks + self._etfs + self._currencies
        for instrument in instruments_list:
            instr = instrument.to_dict()
            instrument_info[instr['figi']] = instr
        return instrument_info

    def _get_candles_by_figi(self, figi, _from, to=None, interval='1min'):
        if to is None:
            to = datetime.now(tz=timezone('Europe/Moscow'))
        candles_payload = self._client.market.market_candles_get(figi, _from, to, interval)

        candles_dict = defaultdict(list)

        for candle in candles_payload.payload.candles:
            for key in candle.to_dict():
                candles_dict[key].append(candle.to_dict()[key])
        df = pd.DataFrame(candles_dict)
        return df

    def get_stock_history(self, ticker, _from=None, to=None, interval='5min', return_plot_figure=True):
        if to is None:
            to = datetime.now(tz=timezone('Europe/Moscow'))
        if _from is None:
            _from = to - timedelta(days=1)
        instr = self._instruments_info[ticker]
        figi = instr['figi']
        candles_df = self._get_candles_by_figi(figi, _from=_from, to=to, interval=interval)
        plot_figure = None
        if return_plot_figure:
            plot_figure = self.get_candle_plot(candles_df, instr)

        return candles_df, plot_figure

    def get_stock_info(self, ticker):
        instr = self._instruments_info[ticker]
        figi = instr['figi']
        now = datetime.now(tz=timezone('Europe/Moscow'))
        yesterday = now - timedelta(days=1)
        candles_df = self._get_candles_by_figi(figi, _from=yesterday, to=now, interval='1min')
        last_price = candles_df.loc[candles_df.time == max(candles_df.time)]['c'].values[-1]
        instr['last_price'] = last_price
        return instr

    def get_candle_plot(self, candles_df, instr_desc):

        fig = go.Figure(data=[go.Candlestick(x=candles_df['time'],
                                             open=candles_df['o'],
                                             high=candles_df['h'],
                                             low=candles_df['l'],
                                             close=candles_df['c'])])
        fig.update_layout(title=instr_desc["name"],
                          xaxis_rangeslider_visible=False,
                          yaxis_title="Price, {}".format(instr_desc['currency']),)

        return fig


class StockPosition(object):
    def __init__(self, position, ops_df=None):
        self._src_position_dict = position.to_dict()
        self.deals_list = None
        if ops_df is not None:
            self.deals_list = self.get_all_deals(ops_df)

    @property
    def figi(self):
        return self._src_position_dict.get("figi")

    @property
    def ticker(self):
        return self._src_position_dict.get("ticker")

    @property
    def num_units(self):
        return self._src_position_dict.get("balance")

    @property
    def blocked(self):
        return self._src_position_dict.get("blocked")

    @property
    def lots(self):
        return self._src_position_dict.get("lots")

    @property
    def avg_price(self):
        return self._src_position_dict.get("average_position_price").get("value")

    @property
    def currency(self):
        return self._src_position_dict.get("average_position_price").get("currency")

    @property
    def name(self):
        return self._src_position_dict.get("name")

    @property
    def instrument_type(self):
        return self._src_position_dict.get("instrument_type")

    def get_all_deals(self, ops_df, exlude_declined=True):
        selected_ops_df = ops_df.loc[ops_df.figi == self.figi]
        selected_ops_df = selected_ops_df.loc[selected_ops_df["status"] == "Done"]
        return selected_ops_df.sort_values(by="date")

    def get_deals_stats(self):
        if self.num_units > 0:
            filtered_deals_list = self.deals_list.loc[self.deals_list.operation_type.isin(["Buy", "Sell", "BuyCard"])]
            global_non_loss_value = np.abs(self.deals_list.payment.sum()) / self.num_units
            # global_non_loss_value = np.abs(filtered_deals_list.payment.sum()) / self.num_units
            return global_non_loss_value, filtered_deals_list

    def __repr__(self):
        non_loss_value, filtered_deals_list = self.get_deals_stats()
        repr_dict = dict()
        repr_dict["instrument_type"] = self.instrument_type
        repr_dict["name"] = self.name
        repr_dict["ticker"] = self.ticker
        repr_dict["lots"] = self.lots
        repr_dict["num_units"] = self.num_units
        repr_dict["avg_price"] = self.avg_price
        repr_dict["currency"] = self.currency
        repr_dict["global_non_loss_value"] = non_loss_value

        return "StockPosition({})".format(json.dumps(repr_dict))


class MyPortfolio:
    def __init__(self, client, start_date="2019-01-01"):
        self._client = client
        self._mi = MarketInfo(client)
        self._start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self._start_date = self._start_date.replace(tzinfo=timezone('Europe/Moscow'))

        self._ops_df = self.get_ops_df()
        self.positions = self.get_my_positions()

    def get_my_positions(self):
        pf = self._client.portfolio.portfolio_get()
        my_positions = dict()

        for position in pf.payload.positions:
            sp = StockPosition(position, self._ops_df)
            my_positions[sp.ticker] = sp
        return my_positions

    def get_ops_df(self):
        to_date = datetime.now(tz=timezone('Europe/Moscow'))
        if self._start_date is not None:
            from_date = self._start_date
        else:
            from_date = to_date - timedelta(days=181)

        ops = self._client.operations.operations_get(_from=from_date.isoformat(),
                                                     to=to_date.isoformat(),
                                                     _request_timeout=30)

        ops_dict = defaultdict(list)
        for op in ops.payload.operations:
            #     ops_df.append()
            op_dict = op.to_dict()
            for key in op_dict:
                if key not in ['trades', 'commission']:
                    ops_dict[key].append(op_dict[key])
                elif key == "trades":
                    deal_sum_from_trade = 0
                    deal_quant_from_trade = 0
                    if op_dict[key] is not None:
                        for trade_dict in op_dict[key]:
                            deal_quant_from_trade += trade_dict['quantity']
                        ops_dict["trade_quantity"].append(deal_quant_from_trade)
                    else:
                        ops_dict["trade_quantity"].append(0)
                elif key == "commission":
                    if op_dict[key] is not None:
                        ops_dict[key + '_currency'].append(op_dict[key]['currency'])
                        ops_dict[key + '_value'].append(op_dict[key]['value'])
                    else:
                        ops_dict[key + '_currency'].append(op_dict['currency'])
                        ops_dict[key + '_value'].append(0)

        ops_df = pd.DataFrame(data=ops_dict)
        instruments_info = self._mi.get_instruments_info_by_figi()
        ops_df["ticker"] = [instruments_info .get(figi).get('ticker') if figi is not None else figi for figi in ops_df.figi]
        ops_df["name"] = [instruments_info .get(figi).get('name') if figi is not None else figi for figi in ops_df.figi]
        return ops_df


class DealsHistory:
    def __init__(self, ticker, ops_df):
        self.ticker = ticker
        self.deals_df = self._get_ops_by_ticker(ops_df)
        self.commission_paid = self._compute_commission()
        self._indexed_deals = self._index_deals()
        self.local_non_loss_value = self.get_local_non_loss_val(self._indexed_deals)
        self.total_profit_value = self.get_total_profit_value(self._indexed_deals)
        self.opened_position_value = self._get_opened_position_value(self._indexed_deals)
        self.dividends = self._compute_dividends()

    def _get_ops_by_ticker(self, ops_df):
        selected_ops_df = ops_df.loc[ops_df.ticker == self.ticker]
        selected_ops_df = selected_ops_df.loc[selected_ops_df["status"] == "Done"]
        return selected_ops_df.sort_values(by="date")

    def _index_deals(self):
        tmp_deals = self.deals_df.loc[self.deals_df["operation_type"].isin(["Buy", "BuyCard", "Sell"])].copy()

        # tmp_deals["deal_idx"] = np.zeros(len(tmp_deals), dtype=int)

        buy_sell = np.array([1 if val in ["Buy", "BuyCard"] else -1 for val in tmp_deals["operation_type"]])

        tmp_deals["signed_quantity"] = buy_sell * tmp_deals["trade_quantity"].values

        deal_idx = 0
        position_quantity = 0
        prev_quantity = 0

        deals_idx = []
        for row in tmp_deals.iterrows():
            position_quantity += row[1]["signed_quantity"]
            deals_idx.append(deal_idx)
            if position_quantity == 0:
                deal_idx += 1

        tmp_deals["deal_idx"] = deals_idx
        return tmp_deals

    def _compute_commission(self):
        return np.abs(self.deals_df.loc[self.deals_df["operation_type"].isin(["BrokerCommission"])].payment).sum()

    def _compute_dividends(self):
        div_sum = np.abs(self.deals_df.loc[self.deals_df["operation_type"].isin(["Dividend"])].payment).sum()
        div_sum = div_sum if div_sum is not None else 0.0
        # div_tax = np.abs(self.deals_df.loc[self.deals_df["operation_type"].isin(["TaxDividend"])].payment).sum()
        # div_tax = div_tax if div_tax is not None else 0.0
        return div_sum

    def _get_last_opened_deals(self, deals_df):

        last_deals = deals_df.loc[deals_df.deal_idx == deals_df.deal_idx.max()]
        if last_deals.signed_quantity.sum() != 0:
            return last_deals

    def get_local_non_loss_val(self, deals_df):
        last_deals = self._get_last_opened_deals(deals_df)
        if last_deals is not None:
            if sum(last_deals.signed_quantity) != 0:
                local_non_loss_val = (np.abs(last_deals.payment.sum()) + self.commission_paid) / \
                                     last_deals.signed_quantity.sum()
                return local_non_loss_val

    def get_total_profit_value_depr(self, deals_df):
        if deals_df.signed_quantity.sum() != 0:
            tmp_deals = deals_df.loc[deals_df.deal_idx != deals_df.deal_idx.max()]
        else:
            tmp_deals = deals_df
        return tmp_deals.payment.sum() - self.commission_paid

    def _get_opened_position_value(self, deals_df):
        last_deals = self._get_last_opened_deals(deals_df)
        if last_deals is not None:
            return last_deals.payment.sum()

    def get_total_profit_value(self, deals_df):
        buy_queue = []
        sell_queue = []
        for idx, deal in deals_df.iterrows():
            # print("{}, {}".format(deal.payment, deal.quantity))
            if deal.operation_type == "Buy":
                buy_queue.append((deal.price, deal.trade_quantity))
            elif deal.operation_type == "Sell":
                sell_queue.append((deal.price, deal.trade_quantity))

        fixed_profit = 0
        while len(buy_queue) > 0 and len(sell_queue) > 0:
            buy_price, buy_quant = buy_queue.pop(0)
            sell_price, sell_quant = sell_queue.pop(0)

            if sell_quant < buy_quant:
                fixed_profit_loc = (sell_price - buy_price) * (sell_quant)
                # print("fp value: leq ", fixed_profit_loc)
                buy_queue.insert(0, (buy_price, buy_quant - sell_quant))
            elif sell_quant > buy_quant:
                fixed_profit_loc = (sell_price - buy_price) * (buy_quant)
                # print("fp value meq: ", fixed_profit_loc)
                sell_queue.insert(0, (sell_price, sell_quant - buy_quant))
            else:
                fixed_profit_loc = (sell_price - buy_price) * (buy_quant)
                # print("fp value eq: ", fixed_profit_loc)
            fixed_profit += fixed_profit_loc

        return fixed_profit
