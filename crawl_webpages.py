import trafilatura
import json
from tqdm import tqdm


with open('data/crawled/url_list.txt', 'r') as f:
    urls = f.read().split('\n')

data = []
for url in tqdm(urls):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        data.append({
            'title': url,
            'url': url,
            'html': text
        })
    except:
        print(f'Failed to download {url}')

with open('data/crawled/webpages.json', 'w') as f:
    json.dump(data, f, indent=4)
    