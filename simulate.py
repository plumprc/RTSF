import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    base = np.linspace(-10, 10, 3000)
    assert len(sys.argv) >= 2 and int(sys.argv[1]) <= 10, 'input #channel'
    channel = int(sys.argv[1])
    
    base = np.linspace(-10, 10, 3000)
    x = []
    for c in range(channel):
        x.append(np.sin((50 - c * 5) * base) + np.random.normal(0, 0.05, 3000))

    x = np.array(x)
    
    # Fake date head. Our methods do not utilize date information
    date_rng = pd.date_range(start='2023/1/1 00:00:00', end='2023/6/15 00:00:00', freq='H')[:3000]
    data = {}
    data['date'] = date_rng
    for idx in range(channel):
        data[str(idx)] = x[idx]

    data['OT'] = data.pop(str(channel-1))
    df = pd.DataFrame(data)
    df.to_csv('multiwave_' + str(channel) + '.csv', index_label=False)
