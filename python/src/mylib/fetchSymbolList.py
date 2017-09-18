import urllib
import pytz
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime

SITE = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SITE = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
START = datetime(1900, 1, 1, 0, 0, 0, 0, pytz.utc)
END = datetime.today().utcnow()

def scrape_list(site):
    hdr = {'User-Agent': 'Mozilla/5.0'}
    page = urllib.request.urlopen(site).read().decode()
    soup = BeautifulSoup(page, "lxml")

    table = soup.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict()
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            sector = str(col[3].string.strip()).lower().replace(' ', '_')
            ticker = str(col[0].string.strip())
            if sector not in sector_tickers:
                sector_tickers[sector] = list()
            sector_tickers[sector].append(ticker)
    return sector_tickers

def getSymbolList():
    sector_tickers = scrape_list(SITE)
    return sector_tickers

#print(sector_tickers)