import gymnasium as gym
from gymnasium import spaces
import xarray as xr
import pandas as pd
import numpy as np
import datetime
import glob
from pathlib import Path    

from collections import Counter
from .utils.history import History
from .utils.portfolio import Portfolio, TargetPortfolio

import tempfile, os
import warnings
warnings.filterwarnings("error")

def basic_reward_function(history: History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]

def dynamic_feature_drawdown(history):
    networth_array = history['portfolio_valuation']
    _max_networth = networth_array[0]
    for networth in networth_array:
        if networth > _max_networth:
            _max_networth = networth
        drawdown = ( networth - _max_networth ) / _max_networth
    return drawdown

def dynamic_feature_max_drawdown(history):
    networth_array = history['portfolio_valuation']
    _max_networth = networth_array[0]
    _max_drawdown = 0
    for networth in networth_array:
        if networth > _max_networth:
            _max_networth = networth
        drawdown = ( networth - _max_networth ) / _max_networth
        if drawdown < _max_drawdown:
            _max_drawdown = drawdown
    return _max_drawdown

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['logs']}
    def __init__(self,
                 ds: xr.Dataset,
                 trade_timeframe: str,
                 positions: list = [0, 1],
                 dynamic_feature_functions=[dynamic_feature_last_position_taken, dynamic_feature_real_position, dynamic_feature_drawdown, dynamic_feature_max_drawdown],
                 reward_function=basic_reward_function,
                 windows=None,
                 trading_fees=0,
                 borrow_interest_rate=0,
                 portfolio_initial_value=1000,
                 initial_position='random',
                 max_episode_duration='max',
                 verbose=1,
                 name="Stock",
                 render_mode="logs"
                 ):
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose
        self.trade_timeframe = trade_timeframe

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position
        assert self.initial_position in self.positions or self.initial_position == 'random', "The 'initial_position' parameter must be 'random' or a position mentioned in the 'position' (default is [0, 1]) parameter."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode
        self._set_ds(ds)
        
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=[self._nb_features]
        )
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=[self.windows, self._nb_features]
            )
        
        self.log_metrics = []

    def _set_ds(self, ds):
        self._features_columns = [var for var in ds.variables if "feature" in var]
        self._info_columns = [var for var in ds.variables if var not in self._features_columns and var not in ["time", "timeframe"]]
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i in range(len(self.dynamic_feature_functions)):
            ds[f"dynamic_feature__{i}"] = xr.DataArray(np.zeros(ds.sizes["time"]), dims="time")
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.ds = ds
        self._obs_array = np.stack([ds[var].values for var in self._features_columns], axis=-1)
        self._info_array = np.stack([ds[var].values for var in self._info_columns], axis=-1)
        self._price_array = ds["close"].values[:, ds["timeframe"].values == self.trade_timeframe]  # Use only the selected timeframe

    def _get_ticker(self, delta=0):
        idx = self._idx + delta
        return {var: self.ds[var].values[idx, self.ds["timeframe"].values == self.trade_timeframe] for var in self._info_columns}

    def _get_price(self, delta=0):
        return self._price_array[self._idx + delta]

    def _get_obs(self):
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self.ds[f"dynamic_feature__{i}"].values[self._idx] = dynamic_feature_function(self.historical_info)

        if self.windows is None:
            _step_index = self._idx
        else:
            _step_index = np.arange(self._idx + 1 - self.windows, self._idx + 1)
        return self._obs_array[_step_index]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._step = 0
        self._position = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
        self._limit_orders = {}
        

        self._idx = 0
        if self.windows is not None: self._idx = self.windows - 1
        if self.max_episode_duration != 'max':
            self._idx = np.random.randint(
                low=self._idx,
                high=len(self.ds["time"]) - self.max_episode_duration - self._idx
            )
        
        self._portfolio = TargetPortfolio(
            position=self._position,
            value=self.portfolio_initial_value,
            price=self._get_price()
        )
        
        self.historical_info = History(max_size=len(self.ds["time"]))
        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.ds["time"].values[self._idx],
            position_index=self.positions.index(self._position),
            position=self._position,
            real_position=self._position,
            data=self._get_ticker(),
            portfolio_valuation=self.portfolio_initial_value,
            portfolio_distribution=self._portfolio.get_portfolio_distribution(),
            reward=0,
        )

        return self._get_obs(), self.historical_info[0]

    def render(self):
        pass

    def _trade(self, position, price=None):
        self._portfolio.trade_to_position(
            position,
            price=self._get_price() if price is None else price,
            trading_fees=self.trading_fees
        )
        self._position = position
        return

    def _take_action(self, position):
        if position != self._position:
            self._trade(position)

    def _take_action_order_limit(self):
        if len(self._limit_orders) > 0:
            ticker = self._get_ticker()
            for position, params in self._limit_orders.items():
                price = self._get_price()
                if position != self._position and params['limit'] <= ticker["high"] and params['limit'] >= ticker["low"]:
                    self._trade(position, price=params['limit'])
                    if not params['persistent']: del self._limit_orders[position]

    def add_limit_order(self, position, limit, persistent=False):
        self._limit_orders[position] = {
            'limit': limit,
            'persistent': persistent
        }

    def step(self, position_index=None):
        if position_index is not None: self._take_action(self.positions[position_index])
        self._idx += 1
        self._step += 1

        self._take_action_order_limit()
        price = self._get_price()
        self._portfolio.update_interest(borrow_interest_rate=self.borrow_interest_rate)
        portfolio_value = self._portfolio.valorisation(price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.ds["time"]) - 1:
            truncated = True
        if isinstance(self.max_episode_duration, int) and self._step >= self.max_episode_duration - 1:
            truncated = True

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.ds["time"].values[self._idx],
            position_index=position_index,
            position=self._position,
            real_position=self._portfolio.real_position(price),
            data=self._get_ticker(),
            portfolio_valuation=portfolio_value,
            portfolio_distribution=portfolio_distribution,
            reward=0
        )
        if not done:
            reward = self.reward_function(self.historical_info)
            self.historical_info["reward", -1] = reward

        if done or truncated:
            self.calculate_metrics()
            self.log()
        return self._get_obs(), self.historical_info["reward", -1], done, truncated, self.historical_info[-1]

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })

    def calculate_metrics(self):
        self.results_metrics = {
            "Market Return": f"{100 * (self.historical_info['data_close', -1] / self.historical_info['data_close', 0] - 1):5.2f}%",
            "Portfolio Return": f"{100 * (self.historical_info['portfolio_valuation', -1] / self.historical_info['portfolio_valuation', 0] - 1):5.2f}%",
        }

        for metric in self.log_metrics:
            self.results_metrics[metric['name']] = metric['function'](self.historical_info)

    def get_metrics(self):
        return self.results_metrics

    def log(self):
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

    def save_for_render(self, dir="render_logs"):
        assert "open" in self.ds and "high" in self.ds and "low" in self.ds and "close" in self.ds, "Your Dataset needs to contain variables: open, high, low, close to render!"
        columns = list(set(self.historical_info.columns) - set([f"date_{col}" for col in self._info_columns]))
        history_df = pd.DataFrame(
            self.historical_info[columns], columns=columns
        )
        history_df.set_index("date", inplace=True)
        history_df.sort_index(inplace=True)
        render_df = self.ds.to_dataframe().join(history_df, how="inner")

        if not os.path.exists(dir):
            os.makedirs(dir)
        render_df.to_pickle(f"{dir}/{self.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")


class MultiDatasetTradingEnv(TradingEnv):
    """
    (Inherits from TradingEnv) A TradingEnv environment that handles multiple datasets.
    It automatically switches from one dataset to another at the end of an episode.
    Bringing diversity by having several datasets, even from the same pair from different exchanges, is a good idea.
    This should help avoiding overfitting.

    It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('MultiDatasetTradingEnv',
            dataset_dir = 'data/*.pkl',
            ...
        )

    :param dataset_dir: A `glob path <https://docs.python.org/3.6/library/glob.html>`_ that needs to match your datasets. All of your datasets needs to match the dataset requirements (see docs from TradingEnv). If it is not the case, you can use the ``preprocess`` param to make your datasets match the requirements.
    :type dataset_dir: str

    :param preprocess: This function takes an xarray.Dataset and returns an xarray.Dataset. This function is applied to each dataset before being used in the environment.

        For example, imagine you have a folder named 'data' with several datasets (formatted as .pkl)

        .. code-block:: python

            import xarray as xr
            import numpy as np
            import gymnasium as gym
            from gym_trading_env

            # Generating features.
            def preprocess(ds: xr.Dataset):
                # You can easily change your inputs this way
                ds["feature_close"] = ds["close"].diff("time") / ds["close"].shift(1, "time")
                ds["feature_open"] = ds["open"] / ds["close"]
                ds["feature_high"] = ds["high"] / ds["close"]
                ds["feature_low"] = ds["low"] / ds["close"]
                ds["feature_volume"] = ds["volume"] / ds["volume"].rolling(time=7*24).max()
                ds = ds.dropna(dim="time")
                return ds

            env = gym.make(
                    "MultiDatasetTradingEnv",
                    dataset_dir= 'examples/data/*.pkl',
                    preprocess= preprocess,
                )

    :type preprocess: function<xarray.Dataset->xarray.Dataset>

    :param episodes_between_dataset_switch: Number of times a dataset is used to create an episode, before moving on to another dataset. It can be useful for performances when `max_episode_duration` is low.
    :type episodes_between_dataset_switch: optional - int
    """

    def __init__(self,
                 dataset_dir,
                 *args,

                 preprocess=lambda ds: ds,
                 episodes_between_dataset_switch=1,
                 **kwargs):
        self.dataset_dir = dataset_dir
        self.preprocess = preprocess
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.dataset_pathes = glob.glob(self.dataset_dir)
        self.dataset_nb_uses = np.zeros(shape=(len(self.dataset_pathes),))
        super().__init__(self.next_dataset(), *args, **kwargs)

    def next_dataset(self):
        self._episodes_on_this_dataset = 0
        # Find the indexes of the less explored dataset
        potential_dataset_pathes = np.where(self.dataset_nb_uses == self.dataset_nb_uses.min())[0]
        # Pick one of them
        random_int = np.random.randint(potential_dataset_pathes.size)
        dataset_path = self.dataset_pathes[random_int]
        self.dataset_nb_uses[random_int] += 1  # Update nb use counts

        self.name = Path(dataset_path).name
        return self.preprocess(xr.open_dataset(dataset_path))

    def reset(self, seed=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self._set_ds(
                self.next_dataset()
            )
        if self.verbose > 1: print(f"Selected dataset {self.name} ...")
        return super().reset(seed)