import os
import requests
from dotenv import load_dotenv

load_dotenv()
r = requests.get("https://content.guardianapis.com/search", params={
    "api-key": os.getenv("GUARDIAN_API_KEY"),
    "section": "world",
    "show-fields": "bodyText",
    "page-size": 5,
})
for a in r.json()["response"]["results"]:
    print(a["webPublicationDate"], "|", a["webTitle"], "|", len(a["fields"]["bodyText"]), "chars")