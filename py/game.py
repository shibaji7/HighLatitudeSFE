import numpy as np
import pandas as pd

def roulette_wheels():
    # Generate random number 'without replacement' between 1 - 36
    values = np.random.choice(np.arange(0, 37), size=37, replace=False)
    # Generate alternate r/b colors
    colors = []
    for i, _ in enumerate(values):
        colors.append("red" if np.mod(i,2)==0 else "black")
    # Assign green to '0'
    colors[np.argmin(values)] = "green"
    wheel = pd.DataFrame()
    wheel["color"], wheel["value"] = colors, values
    return wheel

def martingale(base_money=100, disp=True, color="red"):
    # Create a roulette wheel
    wheel = roulette_wheels()
    # Initialize variable lists for all rounds
    bets, inhands, wins, gains = [], [], [], []
    # Assign base value, initial bet, gain, 
    bet = 1.0
    inhand = np.copy(base_money)
    gain = 0.
    win = 0
    rnd = 1
    if disp: print(f"\n Inital conditions: Bet ${bet} / In hand {inhand} / Bet color {color}")
    # Run the betting skim till 
    # 1. If money in hand is greater than bet
    # 2. Total gain is less than 10.
    while (inhand >= bet) and (np.sum(gains) < 10):
        # Spin the wheel once
        number = np.random.choice(np.arange(0, 37), size=1, replace=True)[0]
        # Know the color of the pit where ball stopped by pervious wheel spin
        ball = wheel[wheel.value==number]
        if ball.iloc[0].color == color:
            win, gain, inhand  = (
                1, 
                np.copy(bet), 
                inhand+np.copy(bet)
            )
        else:
            win, gain, inhand  = (
                0, 
                -1 * np.copy(bet), 
                inhand-np.copy(bet)
            )
        bets.append(bet)
        wins.append(win)
        gains.append(gain)
        inhands.append(inhand)
        if disp: print(f" Round {rnd}: Bet ${bet} / Gain ${gain} / Win {win} / In hand {inhand}")
        rnd += 1
        # Set new bet value (double than previous) for next round lost
        if win == 0:
            bet = 2*np.copy(bet)
    result = pd.DataFrame()
    result["bets"], result["inhands"], result["wins"], result["gains"] = (
        bets, inhands, wins, gains
    )
    final_result = 1 if np.sum(gains) >= 10 else 0
    if disp: print(f" Final result {'Win' if final_result==1 else 'Lost'} / in hand ${inhand} / bet if next round was played ${bet} \n")
    return result, final_result

def simulate_statistics(
    runs = 1000,
    base_money=100, 
    disp=True, 
    color="red"
):
    stats = []
    for _ in range(runs):
        _, finish = martingale(base_money, disp, color)
        stats.append(finish)
    winning_rate = np.sum(stats)/runs
    print(f"\n\n Winning rate {winning_rate*100}%")
    return

simulate_statistics()