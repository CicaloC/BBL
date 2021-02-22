from urllib.request import urlopen
from bs4 import BeautifulSoup as BS
import re


url = 'https://finance.yahoo.com/calendar/earnings/'

page = urlopen(url)

html_bytes = page.read()
html = html_bytes.decode("utf-8")

title_index = html.find("<title>")
start_index = title_index + len("<title>")

end_index = html.find("</title>")

title = html[start_index:end_index]

soup = BS(html, "html.parser")
print(soup.get_text())