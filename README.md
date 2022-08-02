
![logo](https://github.com/thecodeforest/fantasyfootball/blob/main/docs/images/logo.png?raw=true)


-----------------

# Welcome to fantasyfootball
![fantasyfootball](https://github.com/thecodeforest/fantasyfootball/actions/workflows/tests.yml/badge.svg)[![codecov](https://codecov.io/gh/thecodeforest/fantasyfootball/branch/main/graph/badge.svg?token=J2HY3ZOITH)](https://codecov.io/gh/thecodeforest/fantasyfootball)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![Documentation Status](https://readthedocs.org/projects/fantasyfootball/badge/?version=latest)](https://fantasyfootball.readthedocs.io/en/latest/?badge=latest)[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9-blue)](https://badge.fury.io/py/fantasyfootball)[![PyPI version](https://badge.fury.io/py/fantasyfootball.svg)](https://badge.fury.io/py/fantasyfootball)[![License](https://img.shields.io/pypi/l/ansicolortags.svg)](https://img.shields.io/pypi/l/ansicolortags.svg) 



**fantasyfootball** is a Python package that provides up-to-date game data, including player statistics, betting lines, injuries, defensive rankings, and game-day weather data. While many websites offer NFL game data, obtaining it in a format appropriate for analysis or inference requires either (1) a paid subscription or (2) manual weekly downloads with extensive data cleaning. **fantasy football** centralizes game data in a single location while ensuring it is up-to-date throughout the season.

Additionally, **fantasyfootball** streamlines the creation of features for in-season, player-level fantasy point projections. The resulting projections can then determine weekly roster decisions. 

## Installation

```bash
$ pip install fantasyfootball
```

## Benchmarking

The **fantasyfootball** package provides football enthusiasts with the data and tools to create player point projections customized for their league's scoring system. Indeed, a simple comparison between (1) a "naive" projection, and (2) a subscription-based, "industry-grade" projection, revealed that accurate weekly player-level point projections are achievable with **fantasyfootball**. Across all player positions, **fantasyfootball** projections were, on average, 18% more accurate relative to the naive projection (5.6 pts vs. 4.6 pts), while the industry-grade projections were 4% more accurate than the **fantasyfootball** projections (4.6 pts vs. 4.4 pts). The figure below further disaggregates projection performance by player position. More details surrounding this analysis can be found in the [benchmarking notebook](https://github.com/thecodeforest/fantasyfootball/blob/main/examples/benchmarking.ipynb). 

![benchmark](https://github.com/thecodeforest/fantasyfootball/blob/main/docs/images/benchmark_performance_full.png?raw=true)

## Quickstart

Let's walk through an example to illustrate a core use-case of **fantasyfootball**: weekly roster decisions. Imagine it's Tuesday, Week 15 of the 2021 NFL regular season. Your somewhat mediocre team occupies 5th place in the league standings, one spot away from the coveted playoff threshold. It is a must-win week, and you are in the unenviable position of deciding who starts in the Flex roster spot. 
You have three wide receivers available to start, Keenan Allen, Chris Godwin, or Tyler Lockett, and you want to estimate which player will score more points in Week 15. Accordingly, you use the data and feature engineering capabilities in **fantasyfootball** to create player-level point projections. The player with the highest point projection will be slotted into the Flex roster spot, propelling your team to fantasy victory!

Let's start by importing several packages and reading all game data from the 2015-2021 seasons.

```python
from janitor import get_features_targets 
from sklearn.ensemble import RandomForestRegressor

from fantasyfootball.data import FantasyData
from fantasyfootball.features import FantasyFeatures
from fantasyfootball.benchmarking import filter_to_prior_week

# Instantiate FantasyData object for 2015-2021 seasons
fantasy_data = FantasyData(season_year_start=2015, season_year_end=2021)
```
At the time of writing this walkthrough, there are 45 fields available for each player-season-week. For more details on the data, see the [Datasets](#datasets) section below.

Next, we'll create our outcome variable (y) that defines each player's total weekly fantasy points. Then, depending on your league's scoring rules, you can supply standard fantasy football scoring systems, including *yahoo*, *fanduel*, *draftkings*, or create your own *custom* configuration. In the current example, we'll assume you are part of a *yahoo* league with standard scoring.

```python
fantasy_data.create_fantasy_points_column(scoring_source="yahoo")
```

Now that we've added our outcome variable, we'll extract the data and look at a few fields for Tyler Lockett over the past four weeks. Note that a subset of all fields appears below. 

```python
# extract data from fantasy_data object
fantasy_df = fantasy_data.data
# filter to player-season-week in question
lockty_df = fantasy_df.query("name=='Tyler Lockett' & season_year==2021 & 11<=week<=14")   
print(lockty_df)
```

| pid      |   week |   is_away |   receiving_rec |   receiving_td |   receiving_yds |   draftkings_salary |   ff_pts_yahoo |
|:---------|-------:|----------:|----------------:|---------------:|----------------:|--------------------:|---------------:|
| LockTy00 |     11 |         0 |               4 |              0 |             115 |                6000 |           13.5 |
| LockTy00 |     12 |         1 |               3 |              0 |              96 |                6300 |           11.1 |
| LockTy00 |     13 |         0 |               7 |              1 |              68 |                6500 |           16.3 |
| LockTy00 |     14 |         1 |               5 |              1 |             142 |                6700 |           24.7 |

We'll create the feature set that will feed our predictive model in the following section. The first step is to filter to the most recently completed week for all wide receivers (WR). 

```python
# extract the name of our outcome variable
y = fantasy_df.columns[-1]
# filter to all data prior to 2021, Week 15
backtest_df = fantasy_df.filter_to_prior_week(season_year=2021, week_number=14)
# Instantiate FantasyFeatures object for all Wide Receivers
features = FantasyFeatures(backtest_df, position="WR", y=y)   
```

Now, we'll apply a few filters and transformations to prepare our data for modeling: 

* `filter_inactive_games` - Removes games where a player did not play, and therefore recorded zero points.

* `filter_n_games_played_by_season` - Removes players who played only a few games in a season. Setting a threshold is necessary when creating lagged features (which happen to be some of the best predictors of future performance). Removing non-essential players, or those who play only one or two games in a season, also reduces noise and leads to more accurate models. 

* `create_future_week` - Adds leading indicators that we can use to make predictions for Week 15. Recall that, in the current example, we only have game data up to Week 14, so we need to create features for a future, unplayed game. For example, the over/under point projections combined with the point spread estimate how much each team will score. A high-scoring estimate would likely translate into more fantasy points for all players on a team. Another example is weather forecasts. An exceptionally windy game may favor a "run-centric" offense, leading to fewer passing/receiving plays and more rushing plays. Such an occurrence would benefit runnings backs while hurting wide receivers and quarterbacks. 
The *is_future_week* field is also added during this step and allows an easy split between past and future data during the modeling process. 
 
* `add_coefficient_of_variation` - Adds a Coefficient of Variation (CV) for each player based on their past N games. CV shows how much scoring variability there is for each player on a week-to-week basis. While the CV will not serve as an input for predicting player performance in Week 15, it will help us to gauge the consistency of each player when deciding between multiple players. 

* `add_lag_feature` - Add lags of a specified length for lagging indicators, such as receptions, receiving yards, or rushing touchdowns from previous weeks. 

* `add_moving_avg_feature` - Add a moving average of a specified length for lagging indicators.

* `create_ff_signature` - Executes all of the steps used to create "derived features," or features that we've created using some transformation (e.g., a lag or moving average). 

```python

features.filter_inactive_games(status_column="is_active")
features.filter_n_games_played_by_season(min_games_played=1)
features.create_future_week()
features.add_coefficient_of_variation(n_week_window=16)
features.add_lag_feature(n_week_lag=1, lag_columns=y)
features.add_moving_avg_feature(n_week_window=4, window_columns=y)
features_signature_dict = features.create_ff_signature()
```

Having created our feature set, we'll seperate our historical (training) data , denoted `hist_df`, from the future (testing), unplayed game data, denoted `future_df`, using the indicator added above during the `create_future_week` step. 

```python
feature_df = features_signature_dict.get("feature_df")
hist_df = feature_df[feature_df["is_future_week"] == 0]
future_df = feature_df[feature_df["is_future_week"] == 1]
```
For the sake of simplicity, we'll leverage a small subset of raw, untransformed features from our original data, and combine these with the derived features we created in the previous step. 

```python
derived_feature_names = features_signature_dict.get("pipeline_feature_names")
raw_feature_names = ["avg_windspeed", "draftkings_salary"]
all_features = derived_feature_names + raw_feature_names
```

Let's split between our train/hist and test/future data. 
```python
X_hist, y_hist = hist_df[all_features + [y]].get_features_targets(y, all_features)
X_future = future_df[all_features]
```

Now we can fit a simple model and make predictions for the upcoming week. 

```python
rf = RandomForestRegressor(n_estimators=1000, max_depth=4, random_state=0)
rf.fit(X_hist.values, y_hist.values)
y_future = rf.predict(X_future.values).round(1)
```

Below we'll assign our point predictions back to the `future_df` we created for Week 15 and filter to the three players in question.

```python
future_df = future_df.assign(**{f"{y}_pred": y_future})
players = ["Chris Godwin", "Tyler Lockett", "Keenan Allen"]
future_df[["name", "team", "opp", "week", "date", f"{y}_pred"]].query(
    "name in @players"
)
```
| name          | team   | opp   |   week | date       |   ff_pts_yahoo_pred |   cv |
|:--------------|:-------|:------|-------:|:-----------|--------------------:|-----:|
| Keenan Allen  | LAC    | KAN   |     15 | 2021-12-16 |                14.1 |   35 |
| Chris Godwin  | TAM    | NOR   |     15 | 2021-12-19 |                11   |   49 |
| Tyler Lockett | SEA    | LAR   |     15 | 2021-12-21 |                14   |   74 |



Keenan Allen and Tyler Lockett are projected to score ~3 more points than Chris Godwin. And while Keenan Allen and Tyler Lockett have similar projections, over the past 16 games, Allen is much more consistent than Lockett. That is, we should put more faith in Allen's 14-point forecast relative to Lockett. When point projections are equivalent, CV can be a second input when deciding between two players. For example, if the goal is to score many points and win the week, a player with a large CV might be the better option, as they have a higher potential ceiling. In contrast, if the goal is to win, and the total points scored are less critical, then a more consistent player with a small CV is the better option. 



## Datasets
The package provides the following seven datasets by season: 

* **calendar** - The game schedule for the regular season. 

    * `date` - Date (yyyy-mm-dd) of the game
    * `week` - The week of the season
    * `team` - The three letter abbreviation of the team
    * `opp` - The the letter abbreviation of the team's opponent
    * `is_away` - Boolean indicator if the `team` is playing at the opponent's field
    * `season_year` - The year of the season

<br>

* **players** - Each team's roster. Note that only Quarterbacks, Runningbacks, Wide Receivers, and Tight Ends are included. 
    * `name` - A player's first and last name
    * `team` - The three letter abbreviation of the player's team
    * `position` - The two letter abbreviation of the player's position
    * `season_year` - The year of the season

<br>    

* **stats** - The aggregated game statistics for each player. 
    * `pid` - Unique identifier for each player. 
    * `name` - A player's first and last name 
    * `team` - The three letter abbreviation of the player's team
    * `opp` - The three letter abbreviation of the players's weekly opponent 
    * `is_active` - A boolean indicator if the player is active (1) or inactive (0)
    * `date` - Date (yyyy-mm-dd) of the game
    * `result` - The box score of the game, along with an indicator of the outcome (W or L)
    * `is_away` - Boolean indicator if the game was played at the opponent's field (1 = away, 0 = home)
    * `is_start` - Boolean indicator if the player started the game (1 or 0)
    * `g_nbr` - The game number for the player. This differs from the week number, as it accounts for team buy weeks. 
    * `receiving_rec` - The total number of receptions
    * `receiving_yds` - The total number of receiving yards
    * `receiving_td` - The total number of receiving touchdowns
    * `rushing_yds` - The total number or rushing yards
    * `rushing_td` - The total number or rushing yards
    * `receiving_td` - The total number of rushing touchdowns
    * `passing_cmp` - The total number of completed passes
    * `passing_yds` - The total number of passing yards
    * `passing_td` - The total number of passing touchdowns
    * `fumbles_fmb` - The total number of fumbles
    * `passing_int` - The total number of passing interceptions
    * `scoring_2pm` - The total number of 2-point conversions
    * `punt_return_tds` - The total number punt return touchdowns

<br>

* **salary** - The player salaries from DraftKings and FanDuel.
    * `name` - A player's first and last name 
    * `position` - The two letter abbreviation of the player's position
    * `week` - The week of the season
    * `team` - The three letter abbreviation of the team
    * `opp` - The three letter abbreviation of the team's opponent
    * `opp_position_rank` - Opponent rank based on fantasy points allowed by position
    * `fanduel_salary` - Fanduel player salary
    * `draftkings_salary` - DraftKings player salary
    * `season_year` - The year of the season
    

<br>

* **weather** - The game-day weather conditions. Historical data are actual weather conditions, while  data collected prior to a game are based on weather forecasts. Weather data is updated daily throughout the season.
    * `date` - Date (yyyy-mm-dd) of the game
    * `team` - The three letter abbreviation of the team
    * `opp` - The three letter abbreviation of the team's opponent
    * `stadium_name` - Name of the stadium hosting the game. Updated as of 2020. 
    * `roof_type` - Indicates if stadium has dome, retractable roof, or is outdoor. 
    * `is_outdoor` - boolean indicator if stadium is outdoor (1 = is outdoor, 0 = retractable/dome). Note that for stadiums with a retractable roof, it is not possible to determine if roof was open during the game. 
    * `avg_windspeed` - Average daily windspeed (mph)
    * `max_snow_depth` - The maximum depth of the snow (in)
    * `total_precip` - Total daily precipitation (in)
    * `avg_temp` - Average daily temperature (f&deg;). Note that the `avg_temp` of dome/retractable roof stadiums are set to 75&deg;.



<br>


* **betting** - Offensive point projections that are derived from the opening over/under and point-spread. Opening point spreads are refreshed each week on Tuesday.

    * `team` - The three letter abbreviation of team with the point projections
    * `opp` - The three letter abbreviatino of the team's opponent
    * `projected_off_pts` - The projected number of points for each team. For example, if the over/under for a game is 50, and one team is favored to win by 4 points, then the favored team is projected to score 27 points, while the underdog is projected to score 23 points. 
    * `date` - Date (yyyy-mm-dd) of the game
    * `season_year` - The year of the season

<br>

* **defense** - The relative strength of each team's defense along rushing, passing, and receiving. Rankings are updated each week on Tuesday. 

    * `week` - The week of the season
    * `opp` - The three letter team's abbreviation
    * `rushing_def_rank` - Ordinal rank (1 = Best, 32 = Worst) of defensive strength against rushing offense. Combines total rushing yards and total rushing touchdowns to determine strength. The relative weight on each dimension is adjustable. 
    * `receiving_def_rank` - Ordinal rank (1 = Best, 32 = Worst) of defensive strength against receiving offense. Combines total receiving yards and total receiving touchdowns to determine strength. The relative weight on each dimension is adjustable. 
    * `passing_def_rank` - Ordinal rank (1 = Best, 32 = Worst) of defensive strength against passing offense. Combines total passing yards and total passing touchdowns to determine strength. The relative weight on each dimension is adjustable. 
    * `season_year` - The year of the season

<br>

* **injury** - The weekly injury reports for each team. Reports are updated the day before game day.

    * `name` - A player's first and last name 
    * `team` - The three letter abbreviation of team with the point projections
    * `position` - The two letter abbreviation of the player's position
    * `season_year` - The year of the season
    * `week` - The week of the season
    * `injury_type` - The type of injury (e.g., Heel, Hamstring, Knee, Ankle)
    * `has_dnp_tag` - A boolean indicator if the player received a DNP (did not practice) tag  at any point during the week leading up to the game (1) or (0) if not.
    * `has_limited_tag` - A boolean indicator if the player received a Limited tag  at any point during the week leading up to the game (1) or (0) if not.
    * `most_recent_injury_status` - The most recent status for a player prior to game-day (DNP, Limited, Full, no injury).
    * `n_injured` - Indicates the number of injuries reported for a player. For example, "Shin, Ankle" indicates two injuries, or "Shoulder" indicates one injury.

* **draft** - The average draft position (ADP) of each player prior to the season. Positions are based on a standard 12 person draft. 

    * `name` - A player's first and last name 
    * `position` - The two letter abbreviation of the player's position
    * `team` - The three letter abbreviation of team with the point projections    
    * `avg_draft_position` - The average position a player was drafted across many pre-season mock drafts. Players drafted earlier are expected to score more points over a season than those drafted later. 
    * `season_year` - The year of the season

## Data Refresh

While the PyPi version of **fantasyfootball** is updated monthly, the GitHub version is updated weekly during the regular season (Sep 8 - Jan 8). New data is stored in [datasets](https://github.com/thecodeforest/fantasyfootball/tree/main/src/fantasyfootball/datasets/season) directory within the **fantasyfootball** package. If there is a difference between the data in Github and the installed version, creating a `FantasyData` object will download the new data. Note that differences in data persist when a session ends. Updating the installed version of **fantasyfooball** will correct this difference and is recommended at the end of each season. 

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`fantasyfootball` was created by Mark LeBoeuf. It is licensed under the terms of the MIT license.




