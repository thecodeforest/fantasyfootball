
<p align="center">
<img src="docs/images/logo.png" width="1000" height="350">
</p>

-----------------

# Welcome to fantasyfootball
![fantasyfootball](https://github.com/thecodeforest/fantasyfootball/actions/workflows/tests.yml/badge.svg)[![codecov](https://codecov.io/gh/thecodeforest/fantasyfootball/branch/main/graph/badge.svg?token=J2HY3ZOITH)](https://codecov.io/gh/thecodeforest/fantasyfootball)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



**fantasyfootball** is a Python package that provides up-to-date offensive game statistics, betting lines, defensive rankings, and game-day weather data. While many websites provide NFL game data, obtaining it in a format appropriate for analysis or inference requires either (1) a paid subscription or (2) manual weekly downloads with extensive cleaning. **fantasy football** centralizes game data in a single location while also ensuring it is up-to-date throughout the season.

Additionally, **fantasyfootball** streamlines the creation of features for in-season, player-level fantasy point projections. The resulting projections can then determine weekly roster decisions. 

## Why fantasyfootball

Let's walk through a brief, real-world example to illustrate the second use case highlighted above: weekly roster decisions. Imagine it's the Tuesday of Week 10 in the 2021 NFL regular season. Your somewhat mediocre team occupies 5th place, one spot away from the coveted playoff threshold. It is a must-win week, and you are in the unenviable position of deciding who starts in the Flex roster spot. You have two wide-receivers -- Hunter Renfrow, Cole Beasley -- and one running back -- Zack Moss -- for a single roster spot. 
The goal is to start the player who will score the most points in Week 10. To aid in your roster decision, you train a predictive model on past data and then generate point projections for these three players in Week 10. 

```python
from fantasyfootball.data import FantasyData
from fantasyfootball.features import FantasyFeatures

fantasy_data = FantasyData(season_year_start=2018, season_year_end=2021)
fantasy_data.create_fantasy_points_column(scoring_source="draft kings")                           
fantasy_df = fantasy_data.data
fantasy_df.head()
```

Very neat. Much wow. Want to test out your features on a past week?

```python

backtest_df = fantasy_df.filter_to_prior_week(season_year=2021, 
                                              week_number=8
                                              )
features = FantasyFeatures(backtest_df, position="QB")                 
features.filter_n_games_played_by_season(min_games_played=2)                             
features.create_future_week()
# create features lag
yvar = fantasy_df.columns[-1]
features.add_lag_feature(n_week_lag=[1, 2], 
                        lag_columns=["status", yvar]
                                 )
# create moving average feature                                 
fantasy_features.add_moving_avg_feature(n_week_window=[4],
                                        window_columns=["passing_cmp", yvar]
                                        )
feature_names, df_features = fantasy_features.create_ff_signature()                                        

```





## Installation

```bash
$ pip install fantasyfootball
```

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
    * `season_year` - The year of the season
    * `name` - A player's first and last name 
    * `position` - The two letter abbreviation of the player's position
    * `week` - The week of the season
    * `team` - The three letter abbreviation of the team
    * `opp` - The three letter abbreviation of the team's opponent
    * `opp_position_rank` - Opponent rank based on fantasy points allowed by position
    * `fanduel_salary` - Fanduel player salary
    * `draftkings_salary` - DraftKings player salary

<br>

* **weather** - The game-day weather conditions. 
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


* **betting** - Offensive point projections that are derived from the opening over/under and point-spread. Opening point spreads are refreshed the Tuesday of each week.

    * `team` - The three letter abbreviation of team with the point projections
    * `opp` - The three letter abbreviatino of the team's opponent
    * `projected_off_pts` - The projected number of points for each team. For example, if the over/under for a game is 50, and one team is favored to win by 4 points, then the favored team is projected to score 27 points, while the underdog is projected to score 23 points. 
    * `date` - Date (yyyy-mm-dd) of the game
    * `season_year` - The year of the season

<br>

* **defense** - The relative strength of each team's defense along rushing, passing, and receiving. Rankings are updated on Tuesday of each week. 

    * `week` - The week of the season
    * `opp` - The three letter team's abbreviation
    * `rushing_def_rank` - Ordinal rank (1 = Best, 32 = Worst) of defensive strength against rushing offense. Combines total rushing yards and total rushing touchdowns to determine strength. The relative weight on each dimension is adjustable. 
    * `receiving_def_rank` - Ordinal rank (1 = Best, 32 = Worst) of defensive strength against receiving offense. Combines total receiving yards and total receiving touchdowns to determine strength. The relative weight on each dimension is adjustable. 
    * `passing_def_rank` - Ordinal rank (1 = Best, 32 = Worst) of defensive strength against passing offense. Combines total passing yards and total passing touchdowns to determine strength. The relative weight on each dimension is adjustable. 
    * `season_year` - The year of the season


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`fantasyfootball` was created by Mark LeBoeuf. It is licensed under the terms of the MIT license.




