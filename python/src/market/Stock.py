import matplotlib
import matplotlib.pyplot as plt
from src.market.data_helper import get_daily_data

matplotlib.rcParams['examples.directory'] = "D:\\Market"

(x, y) = get_daily_data()

lines = plt.plot(x, y)
plt.xlabel('Time')
plt.ylabel('Price')

plt.title('AAPL')

plt.show()

print("stop")
