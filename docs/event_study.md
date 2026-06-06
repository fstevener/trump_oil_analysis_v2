Für eine saubere wissenschaftliche Implementierung würde ich den Code modular aufbauen und insbesondere folgende Punkte gegenüber vielen Online-Beispielen verbessern:

saubere Zeitzonenbehandlung
Tweets auf die nächste verfügbare Marktminute mappen
Event-Fenster mit vollständigen Beobachtungen erzwingen
Abnormal Returns auf Basis eines Estimation Windows berechnen
mehrere Event-Horizonte testen
t-Test und Wilcoxon
Placebo-/Randomization-Test

Ich nehme an:

tweets.csv
-----------
tweet_id,timestamp,text

wti.csv
-----------
timestamp,open,high,low,close

und dass beide Zeitstempel bereits in UTC vorliegen.

1. Imports
import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_1samp,
    wilcoxon,
    mannwhitneyu
)

import matplotlib.pyplot as plt
2. Parameter

Diese Parameter sollten zentral definiert werden.

# --------------------------------------------------
# Event Study Parameter
# --------------------------------------------------

PRE_EVENT_WINDOW = 30      # Minuten vor Tweet
POST_EVENT_WINDOW = 60     # Minuten nach Tweet

# Schätzung der "normalen" Rendite
ESTIMATION_WINDOW = 120    # Minuten vor Event

# zu untersuchende Horizonte
EVENT_HORIZONS = [5, 15, 30, 60]

# Anzahl Placebo-Events
N_RANDOM_EVENTS = 1000

RANDOM_SEED = 42
3. Daten laden
# --------------------------------------------------
# Tweets
# --------------------------------------------------

tweets = pd.read_csv(
    "tweets.csv",
    parse_dates=["timestamp"]
)

# --------------------------------------------------
# WTI OHLC
# --------------------------------------------------

wti = pd.read_csv(
    "wti.csv",
    parse_dates=["timestamp"]
)

wti = wti.sort_values("timestamp")
wti = wti.set_index("timestamp")
4. Log-Renditen berechnen

Finanzökonomischer Standard:

r
t
	​

=ln(P
t
	​

/P
t−1
	​

)
# --------------------------------------------------
# Log Returns
# --------------------------------------------------

wti["return"] = np.log(
    wti["close"] / wti["close"].shift(1)
)

wti = wti.dropna()
5. Tweet-Zeitpunkte auf Marktindex mappen

Nicht jeder Tweet fällt exakt auf eine vorhandene Marktminute.

Wir mappen daher auf den nächstfolgenden verfügbaren Timestamp.

def map_to_market_timestamp(event_time, market_index):
    """
    Mappt einen Tweet auf die erste verfügbare Marktminute.
    """

    pos = market_index.searchsorted(event_time)

    if pos >= len(market_index):
        return None

    return market_index[pos]
6. Event Window extrahieren

Für jedes Event benötigen wir:

Estimation Window
Event Window
def extract_event_data(
    market_df,
    event_time,
    estimation_window=120,
    pre_window=30,
    post_window=60
):
    """
    Extrahiert die Daten eines einzelnen Events.

    Returns:
        estimation_returns
        event_returns
    """

    idx = market_df.index

    event_timestamp = map_to_market_timestamp(
        event_time,
        idx
    )

    if event_timestamp is None:
        return None

    center_loc = idx.get_loc(event_timestamp)

    estimation_start = (
        center_loc
        - estimation_window
        - pre_window
    )

    estimation_end = (
        center_loc
        - pre_window
        - 1
    )

    event_start = center_loc - pre_window
    event_end = center_loc + post_window

    # Vollständige Fenster erzwingen
    if (
        estimation_start < 0
        or event_end >= len(idx)
    ):
        return None

    estimation_returns = (
        market_df.iloc[
            estimation_start:
            estimation_end + 1
        ]["return"]
        .values
    )

    event_returns = (
        market_df.iloc[
            event_start:
            event_end + 1
        ]["return"]
        .values
    )

    return (
        estimation_returns,
        event_returns
    )
7. Event Study Datensatz erzeugen
event_records = []

for _, tweet in tweets.iterrows():

    result = extract_event_data(
        market_df=wti,
        event_time=tweet["timestamp"],
        estimation_window=ESTIMATION_WINDOW,
        pre_window=PRE_EVENT_WINDOW,
        post_window=POST_EVENT_WINDOW
    )

    if result is None:
        continue

    estimation_returns, event_returns = result

    event_records.append(
        {
            "tweet_id": tweet["tweet_id"],
            "timestamp": tweet["timestamp"],
            "estimation_returns": estimation_returns,
            "event_returns": event_returns
        }
    )

print(f"Valid Events: {len(event_records)}")
8. Abnormal Returns berechnen

Einfaches Market-Adjusted Model:

AR
t
	​

=R
t
	​

−E(R)

mit

E(R)=Mittelwert des Estimation Windows
abnormal_return_matrix = []

for event in event_records:

    expected_return = np.mean(
        event["estimation_returns"]
    )

    abnormal_returns = (
        event["event_returns"]
        - expected_return
    )

    abnormal_return_matrix.append(
        abnormal_returns
    )

abnormal_return_matrix = np.array(
    abnormal_return_matrix
)
9. DataFrame erstellen

Spalten:

-30 ... -1 0 +1 ... +60
relative_minutes = np.arange(
    -PRE_EVENT_WINDOW,
    POST_EVENT_WINDOW + 1
)

abnormal_df = pd.DataFrame(
    abnormal_return_matrix,
    columns=relative_minutes
)

abnormal_df.head()
10. Average Abnormal Return (AAR)
aar = abnormal_df.mean(axis=0)

Plot:

plt.figure(figsize=(12, 6))

plt.plot(
    aar.index,
    aar.values
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.title(
    "Average Abnormal Return Around Trump Tweets"
)

plt.xlabel(
    "Minutes Relative To Tweet"
)

plt.ylabel(
    "Average Abnormal Return"
)

plt.show()
11. Cumulative Average Abnormal Return (CAAR)
caar = aar.cumsum()
plt.figure(figsize=(12, 6))

plt.plot(
    caar.index,
    caar.values
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.title(
    "Cumulative Average Abnormal Return"
)

plt.xlabel(
    "Minutes Relative To Tweet"
)

plt.ylabel(
    "CAAR"
)

plt.show()
12. CAR je Event berechnen

Für die späteren Tests.

car_results = {}

for horizon in EVENT_HORIZONS:

    car = (
        abnormal_df
        .loc[:, 1:horizon]
        .sum(axis=1)
    )

    car_results[horizon] = car
13. Statistische Tests
t-Test
H
0
	​

:CAR=0
for horizon, car in car_results.items():

    t_stat, p_value = ttest_1samp(
        car,
        0
    )

    print("\n")
    print(f"Horizon: {horizon} min")
    print(f"T-stat: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
Wilcoxon

Robust gegen Ausreißer.

for horizon, car in car_results.items():

    stat, p_value = wilcoxon(car)

    print("\n")
    print(f"Horizon: {horizon} min")
    print(f"Wilcoxon p-value: {p_value:.6f}")
14. Placebo Events erzeugen

Wissenschaftlich sehr wichtig.

rng = np.random.default_rng(
    RANDOM_SEED
)

Nur Zeitpunkte zulassen, für die vollständige Fenster existieren.

valid_start = (
    ESTIMATION_WINDOW
    + PRE_EVENT_WINDOW
)

valid_end = (
    len(wti)
    - POST_EVENT_WINDOW
    - 1
)

Zufällige Events ziehen.

random_positions = rng.choice(
    np.arange(valid_start, valid_end),
    size=N_RANDOM_EVENTS,
    replace=False
)
15. Placebo CARs berechnen
placebo_cars = []

for pos in random_positions:

    estimation_returns = (
        wti.iloc[
            pos - ESTIMATION_WINDOW - PRE_EVENT_WINDOW:
            pos - PRE_EVENT_WINDOW
        ]["return"]
        .values
    )

    expected_return = np.mean(
        estimation_returns
    )

    event_returns = (
        wti.iloc[
            pos:
            pos + 15
        ]["return"]
        .values
    )

    abnormal_returns = (
        event_returns
        - expected_return
    )

    placebo_cars.append(
        abnormal_returns.sum()
    )

placebo_cars = np.array(
    placebo_cars
)
16. Vergleich Tweet vs. Placebo

Für 15-Minuten-Horizont.

tweet_car_15 = (
    abnormal_df.loc[:, 1:15]
    .sum(axis=1)
)

Mann-Whitney-Test:

u_stat, p_value = mannwhitneyu(
    tweet_car_15,
    placebo_cars,
    alternative="two-sided"
)

print("\nPlacebo Test")
print("------------")
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value:.6f}")
17. Verteilung visualisieren
plt.figure(figsize=(10, 6))

plt.hist(
    placebo_cars,
    bins=40,
    alpha=0.6,
    density=True,
    label="Random Events"
)

plt.hist(
    tweet_car_15,
    bins=40,
    alpha=0.6,
    density=True,
    label="Trump Tweets"
)

plt.legend()

plt.title(
    "15-Minute CAR Distribution"
)

plt.show()
Interpretation

Ihr habt danach drei zentrale Ergebnisse:

AAR/CAAR-Plots
Gibt es sichtbare Reaktionen um t=0?
t-Test / Wilcoxon
Ist der durchschnittliche CAR signifikant von 0 verschieden?
Placebo-Test
Sind Tweet-Reaktionen größer als normale Marktbewegungen?

Für eine spätere ML-Arbeit würde ich anschließend zusätzlich:

Bootstrap-Konfidenzintervalle für CAR berechnen,
Tweets nach Themen (OPEC, Iran, China, Öl etc.) filtern,
Sentiment modellieren,
und schließlich die Vorhersagekraft von Sentiment gegenüber einer Baseline ohne Tweets evaluieren. Diese Erweiterungen erhöhen die wissenschaftliche Qualität deutlich.
habe diese fehlermeldung nach der Ausführung von Schritt 7 erhalten:

TypeError                                 Traceback (most recent call last)
Cell In[10], line 5
      1 event_records = []
      2 
      3 for _, tweet in tweets.iterrows():
      4 
----> 5     result = extract_event_data(
      6         market_df=wti,
      7         event_time=tweet["timestamp_utc"],
      8         estimation_window=ESTIMATION_WINDOW,

Cell In[8], line 29
     25 
     26     center_loc = idx.get_loc(event_timestamp)
     27 
     28     estimation_start = (
---> 29         center_loc
     30         - estimation_window
     31         - pre_window
     32     )

TypeError: unsupported operand type(s) for -: 'slice' and 'int'

Der Fehler ist sehr aufschlussreich:

TypeError: unsupported operand type(s) for -: 'slice' and 'int'